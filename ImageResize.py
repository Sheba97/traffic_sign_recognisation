from PIL import Image
import os

# Set your input and output folders
for i in range (0,10):
    input_folder = f'complete_data_set/{i}'
    output_folder = f'complete_data_set/{i}_2'

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Desired size
    target_size = (64, 64)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    resized_img.save(output_path)
                    print(f"Resized and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")