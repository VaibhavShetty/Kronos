import argparse
from pathlib import Path

import pandas as pd

from model import Kronos, KronosPredictor, KronosTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Forecast a market CSV with Kronos.")
    parser.add_argument("input", type=Path, help="Input CSV with timestamps and OHLC columns.")
    parser.add_argument("--output", type=Path, default=Path("prediction.csv"))
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--pred-len", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-count", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    required_columns = {"timestamps", "open", "high", "low", "close"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required CSV columns: {missing}")
    if len(df) < args.lookback + args.pred_len:
        raise ValueError("The CSV must contain at least lookback + pred_len rows.")

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    x_df = df.loc[: args.lookback - 1, ["open", "high", "low", "close"]]
    x_timestamp = df.loc[: args.lookback - 1, "timestamps"]
    y_timestamp = df.loc[
        args.lookback : args.lookback + args.pred_len - 1, "timestamps"
    ]

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(
        model,
        tokenizer,
        device=args.device,
        max_context=512,
    )

    prediction = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=args.sample_count,
        verbose=True,
    )
    prediction.to_csv(args.output, index_label="timestamps")
    print(f"Saved {len(prediction)} predictions to {args.output}")


if __name__ == "__main__":
    main()