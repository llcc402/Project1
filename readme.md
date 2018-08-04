# Kaggle API

## Use Kaggle API to download data and submit result

See [github page](https://github.com/Kaggle/kaggle-api) for details of the API.

### How to download and install kaggle api

In order to download and install kaggle api, you must have **pip** installed. We only talk about how to install and use pip in windows (because in Linux it is really easy). 

1. Open anaconda navigator -> Environments. Type "pip" in search box to check whether it is installed. 

2. Install pip if it is not installed.

3. Launch qtconsole or spyder.

4. Use commands `!pip install kaggle` to install kaggle api.

5. Go to <https://www.kaggle.com/username/account> and select 'Create API Token'. This will trgger the download of `kaggle.json`

6. Save `kaggle.json` to folder `C:\Users\<user-name>\.kaggle\`. The trick of creating `.kaggle` in Windows is to write the folder name as `.kaggle.`. (**Do not** forget the dot in the end).

### How to use kaggle api

- Use command `!kaggle competitions download -c competition-name -p folder-name` to download all the files of a competition. 

- Use command `!kaggle competitions submit -f file-name` to submit result.