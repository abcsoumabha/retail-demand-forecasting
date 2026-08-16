import os
import shutil
import kagglehub

def main():
    print("Authenticating with Kaggle...")
    kagglehub.login() 
    
    print("\nDownloading data via KaggleHub...")
    # kagglehub downloads to a hidden cache folder on your machine
    cache_dir = kagglehub.competition_download('m5-forecasting-accuracy')
    
    # Create your local 'data' folder
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCopying files to your local '{output_dir}' folder...")
    
    # Copy the files from the hidden cache into your project's data folder
    for item in os.listdir(cache_dir):
        source = os.path.join(cache_dir, item)
        destination = os.path.join(output_dir, item)
        
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
            print(f"Copied: {item}")
            
    print(f"\nDone! All raw dataset files are now sitting in your '{output_dir}' folder.")

if __name__ == "__main__":
    main()