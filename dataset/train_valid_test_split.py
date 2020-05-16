import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data_dir = os.getenv("DATA_DIR", ".")
    jpvideos_df = pd.read_csv(os.path.join(
        data_dir, "JPvideos.csv"), engine="python")
    use_cols = [
        "title",
        "channel_title",
        "publish_time",
        "tags",
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "comments_disabled",
        "description",
        "category_id"
    ]
    jpvideos_df[use_cols]

    train_df, test_df = train_test_split(jpvideos_df, test_size=0.25)
    train_df, valid_df = train_test_split(train_df, test_size=0.33)
    train_df.to_csv(os.path.join(data_dir, "train.csv"))
    valid_df.to_csv(os.path.join(data_dir, "valid.csv"))
    test_df.to_csv(os.path.join(data_dir, "test.csv"))
