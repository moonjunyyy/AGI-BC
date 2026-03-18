from s2s.runtime import S2SConfig, S2SRuntime


def main() -> None:
    cfg = S2SConfig()
    args = cfg.parse_args()
    print(f"Running with config:\n{args}")
    runtime = S2SRuntime(**args)
    runtime.run()


if __name__ == "__main__":
    main()
