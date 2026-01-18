import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging


class DataSplit:
    """
    Splits the dataset into training and testing sets and saves them to disk.
    """

    def __init__(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, 
                 save_path_train: str = "./data/splits/train.csv", 
                 save_path_test: str = "./data/splits/test.csv", 
                 log_dir: str = "logs"):
        """
        Initializes the DataSplit pipeline.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Seed used by the random number generator.
            save_path_train (str): Path to save the training data CSV.
            save_path_test (str): Path to save the testing data CSV.
            log_dir (str): Directory to store log files.
        """
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path_train), exist_ok=True)
        os.makedirs(os.path.dirname(save_path_test), exist_ok=True)

        # Logger setup
        self.logger = logging.getLogger("DataSplit")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "data_split.log"))
            ch.setLevel(logging.INFO)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.df = df.copy()
        self.test_size = test_size
        self.random_state = random_state
        self.save_path_train = save_path_train
        self.save_path_test = save_path_test
        self.logger.info("DataSplit pipeline initialized.")

    def split_data(self):
        """
        Performs the train-test split.
        
        Assumes 'crop_name_enc' is the target variable (y). 
        Adjust target_column if your target is different.
        """
        self.logger.info("Splitting data into training and testing sets.")
        
        # Define target and features
        # Ensure the target column exists
        target_column = 'crop_name_enc'
        if target_column not in self.df.columns:
            self.logger.error(f"Target column '{target_column}' not found in the DataFrame.")
            raise ValueError(f"Target column '{target_column}' not found.")
            
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        # Perform the split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y # Stratify to maintain class distribution, useful for classification
        )
        
        # Combine features and target for saving
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        self.logger.info(f"Training set shape: {train_df.shape}")
        self.logger.info(f"Testing set shape: {test_df.shape}")
        
        return train_df, test_df

    def save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Saves the train and test DataFrames to CSV files."""
        try:
            train_df.to_csv(self.save_path_train, index=False)
            self.logger.info(f"Training data saved to: {self.save_path_train}")
            
            test_df.to_csv(self.save_path_test, index=False)
            self.logger.info(f"Testing data saved to: {self.save_path_test}")
        except Exception as e:
            self.logger.exception(f"Error saving data splits: {str(e)}")
            raise

    def run(self):
        """Executes the full data splitting pipeline."""
        self.logger.info("Starting data splitting run.")
        train_df, test_df = self.split_data()
        self.save_splits(train_df, test_df)
        self.logger.info("Data splitting completed successfully.")
        return train_df, test_df


if __name__ == "__main__":
    try:
        featured_data_path = "./data/featured/featured_data.csv"
        processed_data_path = "./data/processed/processed_data.csv"
        
        input_path = ""
        if os.path.exists(featured_data_path):
            input_path = featured_data_path
            print(f"Using featured data from: {input_path}")
        elif os.path.exists(processed_data_path):
            input_path = processed_data_path
            print(f"Using processed data from: {input_path}")
        else:
            raise FileNotFoundError("No processed or featured data file found.")

        df_input = pd.read_csv(input_path)
        
        data_splitter = DataSplit(df=df_input, log_dir="./logs")
        train_data, test_data = data_splitter.run()
        
        print("\nData splitting complete.")
    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred during the data splitting process: {e}")