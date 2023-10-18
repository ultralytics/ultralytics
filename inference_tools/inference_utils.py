import csv


def inference_time_csv_writer(results, root, experiment_name):
    # Create a csv file with the following columns: image_name, preprocessing_time, inference_time, postprocessing_time
    with open(root + '/' + experiment_name + '/inference_time.csv', mode='w', newline='') as csv_file:
        # Create csv writer object
        csv_writer = csv.writer(csv_file)

        # Iterate over results
        for result in results:
            # Get name of the image from path and exclude the extension
            image_name = result.path.split('/')[-1].split('.')[0]

            # Write row
            csv_writer.writerow([image_name,
                                 result.speed['preprocess'],
                                 result.speed['inference'],
                                 result.speed['postprocess']])
