# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import os
import re
from typing import Callable, Optional

import yaml

from ultralytics.data.utils import check_det_dataset, verify_image_label
from ultralytics.utils import (
    LOGGER,
)


class DatasetValidation:
    """
    Dataset validation class for check and make a raport about user dataset information and any noticed bugs.

    This class provide core functionality for checking matching files count and dataset structure.

    Attributes:
        dataset (str): path to the dataset.
        yaml (str): path to the yaml file.
        yaml_summary (object): object with info about yaml file.
        errors (List): list with all noticed errors.
        invalid_labels (List): list with all noticed invalid labels.
        is_fix (bool): True if user wants to automatic fix detected bugs.

    Methods:
        check_matching_files_count: Check matching count of image and label files.
        validate: Basic process of validation.
    """

    def __init__(self, dataset, is_fix):
        """
        Initialize DatasetValidation with given configuration and options.

        Args:
            dataset (str): path to the dataset.
            is_fix (bool): True if user wants to automatic fix detected bugs.
        """
        self.dataset = os.path.abspath(dataset)
        self.yaml = None
        self.yaml_summary = {}
        self.errors = []
        self.invalid_labels = []
        self.is_fix = is_fix

    def check_matching_files_count(self, images, labels) -> bool:
        """
        Check matching count of image and label files.

        Args:
            images (str): path to images
            labels (str): path to labels

        Returns:
            (bool): True if images length match to labels length.
        """
        if len(os.listdir(images)) != len(os.listdir(labels)):
            self.errors.append(
                f"Number of images and labels do not match: {len(os.listdir(images))} images, {len(os.listdir(labels))} labels"
            )
            return False
        return True

    def verify_labels(self, labels, yaml_summary="train") -> list:
        """
        Verify labels with images.

        Args:
            labels (List): list with labels.
            yaml_summary (str): subfolder to verify data.

        Returns:
            (List): list of verified labels.
        """
        verify_labels = []
        for image_file in os.listdir(self.yaml_summary[yaml_summary]):
            image_name = os.path.splitext(image_file)[0]
            label_file = image_name + ".txt"

            label_full_path = os.path.join(labels, label_file)
            if os.path.exists(label_full_path):
                image_full_path = os.path.join(self.yaml_summary[yaml_summary], image_file)

                verify_labels.append(
                    verify_image_label(
                        (image_full_path, label_full_path, "", False, self.yaml_summary["nc"], 0, 2, False)
                    )
                )

        return verify_labels

    def validate(self) -> None:
        """Basic process of validation."""
        try:
            if os.path.isdir(self.dataset):
                yamls = glob.glob(os.path.join(self.dataset, "*.yaml"))  # search for any yaml file
                if not yamls:
                    self.errors.append("No YAML file found in the dataset directory.")

                self.yaml = yamls[0]

            structure_validation = check_det_dataset(self.yaml)
            if not isinstance(structure_validation, dict):
                self.errors.append(structure_validation)

            self.yaml_summary = structure_validation

            # get sub directories paths
            labels_path = os.path.join(self.dataset, "labels")
            train_labels = os.path.join(labels_path, "train")
            val_labels = os.path.join(labels_path, "val")

            # check matching count of images and labels in train and val directories
            isTrainImagesMatchingWithLabels = self.check_matching_files_count(self.yaml_summary["train"], train_labels)
            isValImagesMatchingWithLabels = self.check_matching_files_count(self.yaml_summary["val"], val_labels)

            if isTrainImagesMatchingWithLabels and isValImagesMatchingWithLabels:
                verify_labels_structure = []

                # init veryfing image label
                verify_labels_structure.extend(self.verify_labels(train_labels))
                verify_labels_structure.extend(self.verify_labels(val_labels, "val"))

                for el in verify_labels_structure:
                    if isinstance(el, list):
                        invalid_label = el[9:]
                        self.invalid_labels.append(invalid_label)

            if len(self.errors) > 0 and self.is_fix:
                autoFix = AutoFix(self.dataset, self.yaml)
                for error in self.errors:
                    autoFix.fix(error)
            table = Table(self.yaml_summary, self.errors, self.invalid_labels, self.dataset, self.yaml)
            table.draw_summary_table()

        except PermissionError as e:
            self.errors.append(f"Permission denied: {e}")
            LOGGER.error(self.errors)
            return
        except Exception as e:
            self.errors.append(f"Dataset check failed: {e}")
            table = Table(self.yaml_summary, self.errors, self.invalid_labels, self.dataset, self.yaml)
            table.draw_summary_table()
            if self.is_fix:
                autoFix = AutoFix(self.dataset, self.yaml)
                for error in self.errors:
                    autoFix.fix(error)
            return


class AutoFix:
    """
    Fixing noticed bugs in user dataset.

    This class provide core functionality for auto fixing detected bug's.

    Attributes:
        dataset (str): path to user dataset.
        yaml_path (str): path to user yaml file.

    Methods:
        load_yaml: Try to load yaml file.
        save_yaml: Save changes in YAML file.
        fix_missing_yaml: Build basic architecture of YAML file.
        fix_nc_names_mismatch: Changing nc value to names length.
        detect_error: Detect error and fetching correct fix function.
        fix: Initing function to find correct fix function and start process of repairing.
    """

    def __init__(self, dataset, yaml_path):
        """
        Initialize AutoFix with given configuration.

        Args:
            dataset (str): path to user dataset.
            yaml_path (str): path to user yaml file.
        """
        self.dataset = os.path.abspath(dataset)
        self.yaml_path = yaml_path

        self.error_patterns = {
            r"No YAML file found in the dataset directory.": self.fix_missing_yaml,
            r"nc.*does not match.*names|names.*length.*and.*nc.*must match": self.fix_nc_names_mismatch,
        }

    def load_yaml(self) -> bool:
        """
        Try to load yaml file.

        Returns:
            (bool): True if successful loading yaml file.
        """
        if not self.yaml_path or not os.path.exists(self.yaml_path):
            return False

        try:
            with open(self.yaml_path, encoding="utf-8") as file:
                self.yaml_data = yaml.safe_load(file)
            return True
        except Exception as e:
            LOGGER.error(f"Error in loading YAML: {e}")
            return False

    def save_yaml(self, yaml_data=None) -> bool:
        """
        Save changes in YAML file.

        Args:
            yaml_data (Dict): updated or new data.

        Returns:
            (bool): True if successful saving yaml file.
        """
        if not self.yaml_path or not yaml_data:
            return False

        try:
            with open(self.yaml_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(yaml_data, file, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            LOGGER.error(f"Error in YAML saving process: {e}")
            return False

    def fix_missing_yaml(self) -> None:
        """Build basic architecture of YAML file."""
        self.yaml_path = os.path.join(self.dataset, "dataset.yaml")

        yaml_data = {
            "train": os.path.join(self.dataset, "images", "train"),
            "val": os.path.join(self.dataset, "images", "val"),
            "test": os.path.join(self.dataset, "images", "test"),
            "nc": 0,
            "names": {},
        }  # yaml structure
        self.yaml_data = yaml_data
        self.save_yaml(yaml_data)

    def fix_nc_names_mismatch(self) -> None:
        """Changing nc value to names length."""
        if not self.load_yaml():
            return

        if "nc" in self.yaml_data and "names" in self.yaml_data:
            self.yaml_data["nc"] = len(self.yaml_data["names"])  # update yaml nc parameter for currect names length
            self.save_yaml(self.yaml_data)

    def detect_error(self, error_message) -> Optional[Callable[[], None]]:
        """
        Detect error and fetching correct fix function.

        Args:
            error_message (str): noticed error

        Returns:
            Optional[Callable[[], None]]: if fix_function found.
        """
        for pattern, fix_function in self.error_patterns.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                return fix_function

    def fix(self, error) -> None:
        """
        Initing function to find correct fix function and start process of repairing.

        Args:
            error (str): error message.
        """
        LOGGER.info("Fixing...")
        fix_function = self.detect_error(error)
        fix_function() if fix_function else LOGGER.info(f"⚠️ No fix available for error: {error}")
        LOGGER.info("✅ Fixing process completed.")


class Table:
    """
    Table for summary, errors and to show data from dataset check.

    This class provide core functionality for creating table setup and show data on table.

    Attributes:
        yaml_summary (object): summary from yaml validation function.
        errors (List): list of found errors.
        invalid_labels (List): list of founded invalid labels.
        dataset (str): path to dataset.
        yaml (str): path to yaml.

    Methods:
        draw_table: Drawing table schema.
        create_clickable_link: Create clickable link to path where software noticed bug.
        formatting_message: Prepare correct formatting for message.
        draw_summary_table: Drawing finall summary table with presented data.
    """

    def __init__(self, yaml_summary, errors, invalid_labels, dataset, yaml):
        """
        Initialize Table with given configuration and options.

        Args:
            yaml_summary (object): summary from yaml validation function.
            errors (List): list of found errors.
            invalid_labels (List): list of founded invalid labels.
            dataset (str): path to dataset.
            yaml (str): path to yaml.
        """
        self.yaml_summary = yaml_summary
        self.errors = errors
        self.invalid_labels = invalid_labels
        self.dataset = dataset
        self.yaml = yaml

    def draw_table(self, headers, data) -> None:
        """
        Drawing table schema.

        Args:
            headers (List): headers of table UI.
            data (List): data to provide into table UI.
        """
        if not data:
            return

        # calc column widths
        col_widths = [len(str(header)) for header in headers]

        for row in data:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # add padding
        col_widths = [w + 4 for w in col_widths]

        separator = "+" + "+".join(["-" * w for w in col_widths]) + "+"

        LOGGER.info(separator)

        # headers
        header_row = "|"
        for i, header in enumerate(headers):
            header_row += f" {str(header):<{col_widths[i] - 1}}|"
        LOGGER.info(header_row)
        print(separator)

        # data rows
        for row in data:
            data_row = "|"
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cell_str = str(cell)
                    if len(cell_str) > col_widths[i] - 2:
                        cell_str = cell_str[: col_widths[i] - 5] + "..."
                    data_row += f" {cell_str:<{col_widths[i] - 1}}|"
            LOGGER.info(data_row)

        LOGGER.info(separator)

    def create_clickable_link(self, url, text=None) -> str:
        """
        Create clickable link to path where software noticed bug.

        Args:
            url (str): url to file.
            text (str): text to display as a link.
        """
        if text is None:
            text = url
        return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"

    def formatting_message(self, content="") -> str:
        """
        Prepare correct formatting for message.

        Args:
            content (str): content to transform into finall message.

        Returns:
            (str): formatted message.
        """
        path_pattern = r"([C-Z]:[\\\/][^':\"]*)"  # path pattern
        match = re.search(path_pattern, content)  # search for path in content
        message = ""

        if match:
            file_path = match.group(1)
            text_before = content[: match.start()]
            text_after = content[match.end() :]

            clickable_link = self.create_clickable_link(f"file://{file_path}", "📁 See file")
            message = f"{text_before}{clickable_link}{text_after}"
            return message
        return content

    def draw_summary_table(self) -> None:
        """Drawing finall summary table with presented data."""
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("                     DATASET VALIDATION REPORT")
        LOGGER.info("=" * 80)

        if self.yaml_summary:
            LOGGER.info("\n📊 YAML SUMMARY:")
            LOGGER.info("-" * 50)

            yaml_data = []
            for key, value in self.yaml_summary.items():
                if key == "names" and isinstance(value, dict):
                    # Name classes formatting
                    names_str = ", ".join([f"{k}: {v}" for k, v in value.items()])
                    if len(names_str) > 60:
                        names_str = names_str[:60] + "..."
                    yaml_data.append([key.upper(), names_str])
                elif key in ["train", "val", "test"]:
                    # displaying path and file count
                    if os.path.exists(str(value)):
                        file_count = len(os.listdir(str(value)))
                        yaml_data.append([key.upper(), f"{value} ({file_count} files)"])
                    else:
                        yaml_data.append([key.upper(), f"{value} (path not found)"])
                else:
                    yaml_data.append([key.upper(), str(value)])

            self.draw_table(["Parameter", "Value"], yaml_data)

        # add errors and invalid Labels
        LOGGER.info("\n⚠️  VALIDATION ERRORS:")
        LOGGER.info("-" * 50)

        if not self.errors and not self.invalid_labels:
            success_data = [["✅ Status", "No issues found - Dataset is valid!"]]
            self.draw_table(["Result", "Description"], success_data)
        else:
            error_data = []
            error_counter = 1

            # adding errors

            for element in self.errors:
                error_message = self.formatting_message(element)

                error_data.append([f"❌ ERROR {error_counter}", error_message])
                error_counter += 1

            # adding invalid labels
            for invalid_label in self.invalid_labels:
                if isinstance(invalid_label, list) and len(invalid_label) > 0:
                    error_message = self.formatting_message(invalid_label[0])
                    error_data.append([f"🏷️  LABEL {error_counter}", f"Invalid label: {error_message}"])
                    error_counter += 1

            if error_data:
                self.draw_table(["Type", "Description"], error_data)

        LOGGER.info("\n📈 VALIDATION STATISTICS:")
        LOGGER.info("-" * 40)
        LOGGER.info(f"YAML: {self.yaml}")
        stats_data = [
            ["Total Errors", len(self.errors)],
            ["Invalid Labels", len(self.invalid_labels)],
            ["YAML File", "Found" if self.yaml else "Not Found"],
            ["Dataset Path", self.dataset[:60] + "..." if len(self.dataset) > 60 else self.dataset],
        ]

        self.draw_table(["Metric", "Value"], stats_data)

        LOGGER.info("\n" + "=" * 80)
