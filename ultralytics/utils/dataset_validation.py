from ultralytics.data.utils import verify_image_label, check_det_dataset
import glob
import os
import re

class DatasetValidation():
    def __init__(self, dataset):
        self.dataset = os.path.abspath(dataset) 
        self.yaml = None
        self.yaml_summary = {}
        self.elements = []
        self.invaild_labels = []
        
        
    
    def check_matching_files_count(self, images, labels) -> bool:
        if len(os.listdir(images)) != len(os.listdir(labels)):
            self.elements.append(f"Number of images and labels do not match: {len(os.listdir(images))} images, {len(os.listdir(labels))} labels")
            return False
        return True
           

    def validate(self):

        try:
            if os.path.isdir(self.dataset):
                yamls = glob.glob(os.path.join(self.dataset, "*.yaml"))
                if not yamls:
                    self.elements.append("No YAML file found in the dataset directory.")
                    table = Table(self.yaml_summary, self.elements, self.invaild_labels, self.dataset, self.yaml)
                    table.draw_summary_table()
                    return
             
                self.yaml = yamls[0]
               

            structure_validation = check_det_dataset(self.yaml)
            if type(structure_validation) is not dict:
                self.elements.append(structure_validation)
              
                
            
            self.yaml_summary = structure_validation

          
            labels_path = os.path.join(self.dataset, "labels")
            train_labels = os.path.join(labels_path, "train")
            val_labels = os.path.join(labels_path, "val")

            isTrainImagesMatchingWithLabels = self.check_matching_files_count(self.yaml_summary['train'], train_labels)
            isValImagesMatchingWithLabels = self.check_matching_files_count(self.yaml_summary['val'], val_labels)
                    
            if isTrainImagesMatchingWithLabels and isValImagesMatchingWithLabels:
               
                verify_labels_structure = []
                for image_file in os.listdir(self.yaml_summary['train']):
                    for label_file in os.listdir(train_labels):
                        image_full_path = os.path.join(self.yaml_summary['train'], image_file)
                        label_full_path = os.path.join(train_labels, label_file)
                        verify_labels_structure.append(verify_image_label((image_full_path, label_full_path, '', False, self.yaml_summary['nc'], 0, 2, False)))
                
                for el in verify_labels_structure: 
                    if isinstance(el, list):
                        invalid_label = el[9:]
                        self.invaild_labels.append(invalid_label)


            table = Table(self.yaml_summary, self.elements, self.invaild_labels, self.dataset, self.yaml)
            table.draw_summary_table()
                   

        except PermissionError as e:
            self.elements.append(f"Permission denied: {e}")
            print(self.elements)
            return
        except Exception as e:
            self.elements.append(f"Dataset check failed: {e}")
            table = Table(self.yaml_summary, self.elements, self.invaild_labels, self.dataset, self.yaml)
            table.draw_summary_table()
            return


class Table():
    def __init__(self, yaml_summary, elements, invalid_labels, dataset, yaml):
        self.yaml_summary = yaml_summary
        self.elements = elements
        self.invalid_labels = invalid_labels
        self.dataset = dataset
        self.yaml = yaml

    def draw_table(self, headers, data):
        
        if not data:
            return
            
        # calc column widths
        col_widths = [len(str(header)) for header in headers]
        for row in data:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Add padding
        col_widths = [w + 4 for w in col_widths]
        
    
        separator = "+" + "+".join(["-" * w for w in col_widths]) + "+"
        
        print(separator)
        
        # Headers
        header_row = "|"
        for i, header in enumerate(headers):
            header_row += f" {str(header):<{col_widths[i]-1}}|"
        print(header_row)
        print(separator)
        
        # Data rows
        for row in data:
            data_row = "|"
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cell_str = str(cell)
                    if len(cell_str) > col_widths[i] - 2:
                        cell_str = cell_str[:col_widths[i]-5] + "..."
                    data_row += f" {cell_str:<{col_widths[i]-1}}|"
            print(data_row)
        
        print(separator)
    
    def create_clickable_link(self, url, text=None):
        if text is None:
            text = url
        return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"
    
    def formatting_message(self, element):
        path_pattern = r"([C-Z]:[\\\/][^':\"]*)"
        match = re.search(path_pattern, element)
        message = ""
        
        if match:
            file_path = match.group(1)
            text_before = element[:match.start()]
            text_after = element[match.end():]
 
            clickable_link = self.create_clickable_link(f"file://{file_path}", "📁 See file")
            message = f"{text_before}{clickable_link}{text_after}" 
            return message
        return element

    
    def draw_summary_table(self):   
       
        print("\n" + "="*80)
        print("                     DATASET VALIDATION REPORT")
        print("="*80)
        
        
        if self.yaml_summary:
            print("\n📊 YAML SUMMARY:")
            print("-"*50)
            
            yaml_data = []
            for key, value in self.yaml_summary.items():
                if key == 'names' and isinstance(value, dict):
                    # Name classes formatting
                    names_str = ', '.join([f"{k}: {v}" for k, v in value.items()])
                    if len(names_str) > 60:
                        names_str = names_str[:60] + "..."
                    yaml_data.append([key.upper(), names_str])
                elif key in ['train', 'val', 'test']:
                    
                    #displaying path and file count
                    if os.path.exists(str(value)):
                        file_count = len(os.listdir(str(value)))
                        yaml_data.append([key.upper(), f"{value} ({file_count} files)"])
                    else:
                        yaml_data.append([key.upper(), f"{value} (path not found)"])
                else:
                    yaml_data.append([key.upper(), str(value)])
            
            self.draw_table(["Parameter", "Value"], yaml_data)
        
        # Second table: Errors and Invalid Labels
        print("\n⚠️  VALIDATION ERRORS:")
        print("-"*50)
        
        if not self.elements and not self.invalid_labels:
            success_data = [["✅ Status", "No issues found - Dataset is valid!"]]
            self.draw_table(["Result", "Description"], success_data)
        else:
            error_data = []
            error_counter = 1
            
            
            #adding errors
            
            for element in self.elements:
                error_message = self.formatting_message(element)   
                
                error_data.append([
                    f"❌ ERROR {error_counter}",
                    error_message
                ])
                error_counter += 1
            
            #adding invaild labels
            for invalid_label in self.invalid_labels:
                if isinstance(invalid_label, list) and len(invalid_label) > 0:
                   
                    error_message = self.formatting_message(invalid_label[0])
                    error_data.append([
                        f"🏷️  LABEL {error_counter}",
                        f"Invalid label: {error_message}"
                    ])
                    error_counter += 1
            
            if error_data:
                self.draw_table(["Type", "Description"], error_data)
        
        print("\n📈 VALIDATION STATISTICS:")
        print("-"*40)
        print(f"YAML: {self.yaml}")
        stats_data = [
            ["Total Errors", len(self.elements)],
            ["Invalid Labels", len(self.invalid_labels)],
            ["YAML File", "Found" if self.yaml else "Not Found"],
            ["Dataset Path", self.dataset[:60] + "..." if len(self.dataset) > 60 else self.dataset]
        ]
        
        self.draw_table(["Metric", "Value"], stats_data)
        
        print("\n" + "="*80)
         