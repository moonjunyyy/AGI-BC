import torch
from transformers import pipeline, AutoModelForCausalLM
from transformers import AutoTokenizer;
MODEL = 'beomi/KoAlpaca-Polyglot-5.8B'


class KoAlpaca:
    def __init__(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device=f"cuda", non_blocking=True)
        self.model.eval()

        self.pipe = pipeline(
            'text-generation', 
            model=self.model,
            tokenizer=MODEL,
            device=0
        )

    def ask(self, x, is_input_full=False):
        ans = self.pipe(
            x, 
            do_sample=True, 
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
            eos_token_id=2,
            pad_token_id=0,
        )
        print(ans[0]['generated_text'])