import os
import pandas as pd
import cv2

def prepare_lol(path):
    df = pd.DataFrame()
    for data_type in os.listdir(path):
        low_light_images = []
        normal_light_images = []
        
        low_light_dir = os.path.join(path, data_type, "low")
        normal_light_dir = os.path.join(path, data_type, "high")

        for low_light_file_name, normal_light_file_name in zip(os.listdir(low_light_dir), os.listdir(normal_light_dir)):
            if os.path.splitext(low_light_file_name)[1] not in [".jpg", ".png"] or os.path.splitext(normal_light_file_name)[1] not in [".jpg", ".png"]:
                continue
            low_light_images.append(os.path.join(low_light_dir, low_light_file_name))
            normal_light_images.append(os.path.join(normal_light_dir, normal_light_file_name))

        df_section = pd.DataFrame({
            "low_light_path": low_light_images,
            "normal_light_path": normal_light_images,
            "section": data_type,
        })
        
        df = pd.concat([df, df_section], ignore_index=True) if df is not None else df_section
        
    df["low_light"] = df["low_light_path"].apply(lambda x: os.path.split(x)[-1])
    df["normal_light"] = df["normal_light_path"].apply(lambda x: os.path.split(x)[-1])
    df = df[["low_light", "low_light_path", "normal_light", "normal_light_path", "section"]]
    
    file_name = "lol_paired_with_filename.csv"
    output_path = os.path.join(path, file_name)
    df.to_csv(output_path, index=False)

def prepare_lolv2(path):
    
    def check_number_pattern(df):
        def extract_number(filename):
            name = os.path.splitext(filename)[0]
            parts = name.split('low')
            if len(parts) > 1:
                return parts[1]
            parts = name.split('normal')
            if len(parts) > 1:
                return parts[1]
            parts = name.split('r')
            if len(parts) > 1:
                return parts[1][:-1]
            return None
    
        filter_mask = df['low_light'].apply(extract_number) != df['normal_light'].apply(extract_number)
        return filter_mask
    
    
    df = pd.DataFrame()
    for data_type in os.listdir(path):
        for section in os.listdir(os.path.join(path, data_type)):
            low_light_images = []
            normal_light_images = []
            
            low_light_dir = os.path.join(path, data_type, section, "low")
            normal_light_dir = os.path.join(path, data_type, section, "normal")

            for low_light_file_name, normal_light_file_name in zip(os.listdir(low_light_dir), os.listdir(normal_light_dir)):
                if os.path.splitext(low_light_file_name)[1] not in [".jpg", ".png"] or os.path.splitext(normal_light_file_name)[1] not in [".jpg", ".png"]:
                    continue
                low_light_images.append(os.path.join(low_light_dir, low_light_file_name))
                normal_light_images.append(os.path.join(normal_light_dir, normal_light_file_name))

            df_section = pd.DataFrame({
                "low_light_path": low_light_images,
                "normal_light_path": normal_light_images,
                "data_type": data_type,
                "section": section
            })
            df = pd.concat([df, df_section], ignore_index=True) if df is not None else df_section
            
    
    df["low_light"] = df["low_light_path"].apply(lambda x: os.path.split(x)[-1])
    df["normal_light"] = df["normal_light_path"].apply(lambda x: os.path.split(x)[-1])
    df = df[["low_light", "low_light_path", "normal_light", "normal_light_path", "data_type", "section"]]
    filter_mask = check_number_pattern(df)
    df[filter_mask]

    file_name = "lolv2_paired_with_filename.csv"
    output_path = os.path.join(path, file_name)
    df.to_csv(output_path, index=False)

def main(data_path):
    prepare_lol(data_path["lol"])
    prepare_lolv2(data_path["lolv2"])
    

if __name__ == "__main__":
    data_path = {
        "lol":"Our_CycleGAN\Dataset\LOL",
        "lolv2":"Our_CycleGAN\Dataset\LOL-v2"
    }

    main(data_path)