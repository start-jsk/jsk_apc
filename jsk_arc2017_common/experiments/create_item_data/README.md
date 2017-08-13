# create_item_data

![](https://user-images.githubusercontent.com/4310419/29253811-2187f0b2-80c2-11e7-8453-df047624af7b.png)

## Example

```bash
mkdir -p item_data/Noodle_Seafood
# ex. photos taken by iPhone
mv ~/Downloads/Photos.zip item_data/Noodle_Seafood
# item_data/Noodle_Seafood/Photos.zip -> item_data/Noodle_Seafood/*.jpg
(cd item_data/Noodle_Seafood && unzip Photos.zip && rm item_data/Photos.zip)
./01_create_info_json.py
./02_organize_image_files.py
./03_annotate_image_files.py
```


## Usage

```bash
git clone https://github.com/wkentaro/mvtk.git
(cd mvtk && ./install.sh)

./view_item_data.py item_data
```
