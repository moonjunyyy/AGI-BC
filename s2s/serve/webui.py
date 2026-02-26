"""
Gradio web UI for S2S inference.

Launch: python -m s2s.serve.webui --server-url ws://localhost:8998
"""
import argparse
import queue
import threading
import time
from typing import Optional

import gradio as gr
import numpy as np
import torch


def create_ui(server_url: str = "ws://localhost:8998"):
    """Create and return the Gradio interface."""
    import websocket as ws_lib  # websocket-client
    import json

    chat_history = []

    def _connect_and_stream(audio_data, model_choice, temperature, max_tokens, history):
        """Send audio to server, get text+audio response."""
        if audio_data is None:
            return history, None

        sr, samples = audio_data
        # Convert to 16-bit PCM bytes
        if samples.dtype != np.int16:
            samples = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = samples.tobytes()

        # Connect via WebSocket
        received_text = []
        received_audio = []
        done_event = threading.Event()

        ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = ws_url.rstrip("/") + "/ws/chat"

        def on_message(wsapp, msg):
            if isinstance(msg, bytes):
                # PCM audio
                arr = np.frombuffer(msg, dtype=np.int16)
                received_audio.append(arr)
            else:
                try:
                    data = json.loads(msg)
                    if data.get("type") == "text":
                        received_text.append(data.get("text", ""))
                    elif data.get("error"):
                        received_text.append(f"[Error: {data['error']}]")
                except Exception:
                    pass

        def on_close(wsapp, *args):
            done_event.set()

        def on_error(wsapp, err):
            received_text.append(f"[WebSocket error: {err}]")
            done_event.set()

        wsapp = ws_lib.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_close=on_close,
            on_error=on_error,
        )

        def _run():
            wsapp.run_forever()

        wst = threading.Thread(target=_run, daemon=True)
        wst.start()
        time.sleep(0.2)  # Wait for connection

        # Send audio chunks
        chunk_size = 1920 * 2  # 1920 samples * 2 bytes
        for offset in range(0, len(pcm_bytes), chunk_size):
            wsapp.send(pcm_bytes[offset:offset + chunk_size], opcode=ws_lib.ABNF.OPCODE_BINARY)

        # Signal end
        wsapp.send("END")

        # Wait for response (max 10 seconds)
        done_event.wait(timeout=10.0)
        wsapp.close()

        text = " ".join(received_text)
        history = list(history or [])
        history.append(("(audio input)", text if text else "(no text response)"))

        # Combine audio
        audio_out = None
        if received_audio:
            combined = np.concatenate(received_audio)
            audio_out = (24000, combined)

        return history, audio_out

    # -----------------------------------------------------------------------
    # Direct inference UI (no server required)
    # -----------------------------------------------------------------------

    def _direct_infer(audio_data, model_choice, temperature, max_tokens, history, model_state):
        """Direct inference without server."""
        model = model_state.get("model") if model_state else None
        if model is None:
            history = list(history or [])
            history.append(("(audio)", "[Error: no model loaded. Use server mode or load a model.]"))
            return history, None, model_state

        if audio_data is None:
            return history, None, model_state

        sr, samples = audio_data
        if samples.dtype != np.int16:
            samples = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        waveform = torch.from_numpy(samples.astype(np.float32) / 32768.0).unsqueeze(0).unsqueeze(0)

        results = list(model.generate_stream(iter([waveform]), temperature=temperature, max_new_tokens=max_tokens))
        text = " ".join(r.get("text", "") for r in results)
        audio_tensors = [r["audio"] for r in results if r.get("audio") is not None]

        history = list(history or [])
        history.append(("(audio input)", text or "(no text)"))

        audio_out = None
        if audio_tensors:
            combined = torch.cat(audio_tensors, dim=-1).squeeze().numpy()
            combined_i16 = (combined * 32767).clip(-32768, 32767).astype(np.int16)
            audio_out = (24000, combined_i16)

        return history, audio_out, model_state

    # -----------------------------------------------------------------------
    # Gradio blocks
    # -----------------------------------------------------------------------

    with gr.Blocks(title="S2S Web UI") as demo:
        gr.Markdown("# S2S: Speech-to-Speech Demo")
        model_state = gr.State({})

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column():
                    mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Microphone Input")
                    model_choice = gr.Dropdown(
                        choices=["omni2", "moshi"], value="omni2", label="Model"
                    )
                    temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Temperature")
                    max_tokens = gr.Slider(64, 1024, value=256, step=64, label="Max New Tokens")
                    submit_btn = gr.Button("Submit", variant="primary")
                    use_server = gr.Checkbox(value=True, label="Use server (ws://localhost:8998)")
                with gr.Column():
                    chat_box = gr.Chatbot(label="Dialogue")
                    audio_output = gr.Audio(label="Response Audio", autoplay=True)

            def _submit(audio, choice, temp, maxt, hist, state, use_srv):
                if use_srv:
                    hist, audio_out = _connect_and_stream(audio, choice, temp, maxt, hist)
                    return hist, audio_out, state
                else:
                    return _direct_infer(audio, choice, temp, maxt, hist, state)

            submit_btn.click(
                _submit,
                inputs=[mic_input, model_choice, temperature, max_tokens, chat_box, model_state, use_server],
                outputs=[chat_box, audio_output, model_state],
            )

        with gr.Tab("Dual-Agent Eval"):
            gr.Markdown("## Dual-Agent Evaluation")
            with gr.Row():
                goal_desc = gr.Textbox(label="Goal Description", placeholder="e.g., reach agreement on the best color")
                keywords = gr.Textbox(label="Keywords (comma-separated)", placeholder="e.g., agree,deal,yes")
                max_turns = gr.Slider(2, 50, value=10, step=1, label="Max Turns")
            run_eval_btn = gr.Button("Run Evaluation")
            eval_output = gr.JSON(label="Evaluation Result")

            def _run_eval(goal_text, kw_text, max_t, state):
                model = state.get("model") if state else None
                if model is None:
                    return {"error": "No model loaded"}, state
                from ..pipeline.eval_dialogue import DualAgentEvaluator, DialogueGoal
                keywords_list = [k.strip() for k in kw_text.split(",") if k.strip()]
                goal = DialogueGoal(description=goal_text, keywords=keywords_list, max_turns=int(max_t))
                evaluator = DualAgentEvaluator(model, model, goal)  # Same model for both agents
                result = evaluator.run()
                return {
                    "turns": result.turns,
                    "done_reason": result.done_reason,
                    "goal_achieved": result.goal_achieved,
                    "transcript": result.transcript,
                }, state

            run_eval_btn.click(
                _run_eval,
                inputs=[goal_desc, keywords, max_turns, model_state],
                outputs=[eval_output, model_state],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="S2S Web UI")
    parser.add_argument("--server-url", default="ws://localhost:8998")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_ui(server_url=args.server_url)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
