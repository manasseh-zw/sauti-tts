#!/usr/bin/env python3
"""
Generate dataset-profile charts and summary artifacts for the Sauti TTS report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import soundfile as sf


def _load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    df["duration"] = df["duration"].astype(float)
    df["speaker_id"] = df["speaker_id"].astype(str)
    df["char_count"] = df["text"].fillna("").str.len()
    df["word_count"] = df["text"].fillna("").str.split().str.len()
    return df


def _summarize_metadata(df: pd.DataFrame) -> dict:
    speaker_stats = (
        df.groupby("speaker_id")
        .agg(
            utterances=("id", "count"),
            total_seconds=("duration", "sum"),
            mean_duration=("duration", "mean"),
            gender=("gender", "first"),
        )
        .sort_values("total_seconds", ascending=False)
        .reset_index()
    )
    speaker_stats["total_minutes"] = speaker_stats["total_seconds"] / 60.0

    return {
        "num_utterances": int(len(df)),
        "total_hours": round(float(df["duration"].sum() / 3600.0), 3),
        "min_duration_sec": round(float(df["duration"].min()), 3),
        "max_duration_sec": round(float(df["duration"].max()), 3),
        "mean_duration_sec": round(float(df["duration"].mean()), 3),
        "median_duration_sec": round(float(df["duration"].median()), 3),
        "mean_word_count": round(float(df["word_count"].mean()), 2),
        "median_word_count": round(float(df["word_count"].median()), 2),
        "mean_char_count": round(float(df["char_count"].mean()), 2),
        "median_char_count": round(float(df["char_count"].median()), 2),
        "num_speakers": int(df["speaker_id"].nunique()),
        "gender_counts": {
            str(k): int(v)
            for k, v in df["gender"].value_counts(dropna=False).sort_index().items()
        },
        "speaker_stats": [
            {
                "speaker_id": row["speaker_id"],
                "utterances": int(row["utterances"]),
                "total_seconds": round(float(row["total_seconds"]), 3),
                "total_minutes": round(float(row["total_minutes"]), 3),
                "mean_duration_sec": round(float(row["mean_duration"]), 3),
                "gender": str(row["gender"]),
            }
            for _, row in speaker_stats.iterrows()
        ],
    }


def _load_output_audio(output_dir: Path) -> pd.DataFrame:
    records = []
    if not output_dir.exists():
        return pd.DataFrame(columns=["name", "duration_sec", "sample_rate", "channels"])

    for wav_path in sorted(output_dir.glob("*.wav")):
        info = sf.info(str(wav_path))
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        records.append(
            {
                "name": wav_path.name,
                "duration_sec": duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
            }
        )
    return pd.DataFrame(records)


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_duration_histogram(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(df["duration"], bins=24, color="#0f766e", edgecolor="white")
    plt.title("Sauti TTS Prepared Utterance Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Utterance count")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_speaker_minutes(df: pd.DataFrame, path: Path) -> None:
    speaker_minutes = (
        df.groupby("speaker_id")["duration"].sum().sort_values(ascending=False) / 60.0
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=speaker_minutes.index,
        y=speaker_minutes.values,
        palette="crest",
        hue=speaker_minutes.index,
        legend=False,
    )
    plt.title("Prepared Speech Duration by Speaker")
    plt.xlabel("Speaker ID")
    plt.ylabel("Total duration (minutes)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_gender_counts(df: pd.DataFrame, path: Path) -> None:
    gender_counts = df["gender"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(7, 4.5))
    sns.barplot(
        x=gender_counts.index,
        y=gender_counts.values,
        palette=["#1d4ed8", "#15803d"],
        hue=gender_counts.index,
        legend=False,
    )
    plt.title("Prepared Utterances by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Utterance count")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_text_length_vs_duration(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8.5, 5.5))
    sns.scatterplot(
        data=df,
        x="word_count",
        y="duration",
        hue="gender",
        alpha=0.7,
        palette="Set2",
        s=35,
    )
    plt.title("Text Length vs. Audio Duration")
    plt.xlabel("Word count")
    plt.ylabel("Duration (seconds)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_output_durations(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(11, 5.5))
    sns.barplot(
        data=df,
        x="name",
        y="duration_sec",
        palette="mako",
        hue="name",
        legend=False,
    )
    plt.title("Available Generated Sample Durations")
    plt.xlabel("Output file")
    plt.ylabel("Duration (seconds)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Sauti TTS report assets.")
    parser.add_argument(
        "--metadata",
        default="tmp/train_metadata.csv",
        help="Prepared training metadata CSV.",
    )
    parser.add_argument(
        "--outputs",
        default="outputs",
        help="Directory containing generated audio samples.",
    )
    parser.add_argument(
        "--asset-dir",
        default="reports/assets",
        help="Directory for generated charts and JSON summaries.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    outputs_path = Path(args.outputs)
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    df = _load_metadata(metadata_path)
    summary = _summarize_metadata(df)
    output_audio_df = _load_output_audio(outputs_path)

    _save_json(asset_dir / "dataset_profile_summary.json", summary)
    _save_json(
        asset_dir / "output_audio_inventory.json",
        {
            "num_files": int(len(output_audio_df)),
            "files": [
                {
                    "name": row["name"],
                    "duration_sec": round(float(row["duration_sec"]), 3),
                    "sample_rate": int(row["sample_rate"]),
                    "channels": int(row["channels"]),
                }
                for _, row in output_audio_df.iterrows()
            ],
        },
    )

    _plot_duration_histogram(df, asset_dir / "dataset_duration_histogram.png")
    _plot_speaker_minutes(df, asset_dir / "speaker_duration_minutes.png")
    _plot_gender_counts(df, asset_dir / "gender_distribution.png")
    _plot_text_length_vs_duration(df, asset_dir / "text_length_vs_duration.png")
    if not output_audio_df.empty:
        _plot_output_durations(output_audio_df, asset_dir / "output_sample_durations.png")

    print(f"Wrote assets to {asset_dir}")


if __name__ == "__main__":
    main()
