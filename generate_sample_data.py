"""
Generate Sample Data Script
This script generates a sample of the Titanic dataset for quick testing
"""

import pandas as pd
import seaborn as sns


def generate_sample_data(
    n_samples: int = 100, output_path: str = "data/sample_titanic.csv"
):
    """
    Generate a sample of the Titanic dataset

    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the sample data
    """
    print(f"Loading Titanic dataset from scikit-learn (via seaborn)...")

    # Load full Titanic dataset
    df = sns.load_dataset("titanic")

    print(f"Full dataset shape: {df.shape}")
    print(f"\nDataset info:")
    print(df.info())

    print(f"\nFirst few rows:")
    print(df.head())

    print(f"\nSurvival statistics:")
    print(df["survived"].value_counts())

    # Generate stratified sample
    sample_df = df.groupby("survived", group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_samples // 2), random_state=42)
    )

    # Save sample
    sample_df.to_csv(output_path, index=False)
    print(f"\nSample data ({len(sample_df)} rows) saved to: {output_path}")

    print(f"\nSample survival statistics:")
    print(sample_df["survived"].value_counts())

    return sample_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample Titanic data")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_titanic.csv",
        help="Output path (default: data/sample_titanic.csv)",
    )

    args = parser.parse_args()

    generate_sample_data(args.samples, args.output)
