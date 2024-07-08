#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact.

Author: Zidane
Date: 08-07-2024
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    """
    Main function to execute the data cleaning process.

    Parameters:
    args (argparse.Namespace): The command line arguments.
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading input artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Read the dataset
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info("Dropping outliers")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save the results to a CSV file
    logger.info("Saving cleaned data to CSV")
    df.to_csv("clean_sample.csv", index=False)

    # Upload the cleaned dataset to W&B
    logger.info("Uploading cleaned data to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact to clean",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output cleaned artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to filter outliers",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to filter outliers",
        required=True
    )

    args = parser.parse_args()

    go(args)
