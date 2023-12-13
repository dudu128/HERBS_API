## 從github clone下之後，下載pretrained weight(因為github限制超過100MB)，並放到train_herbs_api->code->pretrained內(可File Structure)
在terminal輸入
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/u/0/uc?id=1bxDjd1VZjzc1jAbi8JOmZ_AyoHBFgHpO&export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')" -O "emb_func_best.pth" && rm -rf /tmp/cookies.txt

## 建立image，在terminal輸入
docker build -t herbs_train_image .

## 建立container，在terminal輸入
docker run -idt --gpus all --shm-size 8G --name herbs_train_container -p 8008:8008 herbs_train_image

## 在網址欄中輸入就可以進到api網頁
server:8008/docs  
e.g. http://hc7.isl.lab.nycu.edu.tw:8008/docs

## 下載skd pretrained weight並放到docker code/pretrained資料  
https://drive.google.com/drive/folders/1RyxQsaPoh5XDhIP5OMODnSg9oSLff-o_?usp=sharing  
下載emb_func_best.pth  
docker cp /本地/路徑/檔案或目錄 容器ID或容器名稱:/容器內/目標路徑  
e.g. docker cp /hcds_vol/private/NCU/duncan/temp/pretrained/emb_func_best.pth herbs_train_container:/code/pretrained


## 查看logs，在terminal輸入
docker logs  herbs_train_container
