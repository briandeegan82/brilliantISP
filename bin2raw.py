import struct
import csv
import os

# Constants
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1536
FRAME_SIZE = 6001936  # 8 (SoF) + 8 (FSIN) + 6001920 (Image data)
IMAGE_DATA_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 2  # 2 bytes per pixel

# Input and Output Directories
input_directory = '/media/brian/ssd-drive/drive/input_camera_data'
base_output_directory = '/media/brian/ssd-drive/drive/output'

# Ensure the base output directory exists
if not os.path.exists(base_output_directory):
    os.makedirs(base_output_directory)
    print(f"Created base output directory: {base_output_directory}")

# Get all .bin files from the input directory
bin_files = [f for f in os.listdir(input_directory) if f.endswith('.bin')]

if not bin_files:
    print(f"No .bin files found in {input_directory}. Exiting.")
else:
    for bin_filename in bin_files:
        full_bin_path = os.path.join(input_directory, bin_filename)
        
        # Create a unique output folder for each .bin file
        # The folder name will be derived from the .bin filename
        output_folder_name = os.path.splitext(bin_filename)[0] + "_extracted"
        current_output_folder = os.path.join(base_output_directory, output_folder_name)
        
        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)
            print(f"Created output folder for {bin_filename}: {current_output_folder}")

        csv_filename = os.path.join(current_output_folder, f'{os.path.splitext(bin_filename)[0]}_metadata.csv')

        print(f"\n--- Processing {bin_filename} ---")

        try:
            with open(full_bin_path, 'rb') as bin_file:
                frame_index = 0
                saved_frame_count = 0

                with open(csv_filename, 'w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(['Frame Index', 'Output Filename', 'SoF Timestamp', 'FSIN Timestamp'])

                    while True:
                        frame_data = bin_file.read(FRAME_SIZE)

                        if not frame_data:
                            break

                        if len(frame_data) < FRAME_SIZE:
                            print(f"Warning: Incomplete frame found at the end of {bin_filename}. Skipping remaining data.")
                            break

                        # Extract FSIN Timestamp (8 bytes)
                        fsin_timestamp = struct.unpack('>Q', frame_data[:8])[0]  # Assuming big-endian

                        # Extract SoF Timestamp (8 bytes)
                        sof_timestamp = struct.unpack('>Q', frame_data[8:16])[0]  # Assuming big-endian

                        # Extract Image Data (6001920 bytes)
                        image_data = frame_data[16:16 + IMAGE_DATA_SIZE]

                        # Only save every 10th frame
                        if frame_index % 10 == 0:
                            output_image_path = os.path.join(current_output_folder, f'frame_{frame_index:04d}_fsin_{fsin_timestamp}.raw')
                            with open(output_image_path, 'wb') as img_file:
                                img_file.write(image_data)

                            csv_writer.writerow([frame_index, output_image_path, sof_timestamp, fsin_timestamp])

                            print(f'Saved {output_image_path} (SoF: {sof_timestamp}, FSIN: {fsin_timestamp})')
                            saved_frame_count += 1

                        frame_index += 1
            print(f"Extraction for {bin_filename} complete. Saved {saved_frame_count} frames. Metadata saved to {csv_filename}")

        except FileNotFoundError:
            print(f"Error: The file {full_bin_path} was not found.")
        except Exception as e:
            print(f"An error occurred while processing {bin_filename}: {e}")

print("\nAll specified .bin files have been processed.")
