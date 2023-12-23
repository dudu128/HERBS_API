## 從github clone下，到cd到train_herbs_api->code路徑  
```
cd /HERBS_API/train_herbs_api/code
```

## 在train_herbs_api->code->pretrained內下載pretrained weight(因為github限制超過100MB)  
先cd到train_herbs_api->code->pretrained  
```
cd pretrained
```

## 在terminal輸入，會得到emb_func_best.pth(pretrained weight)  
```
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/u/0/uc?id=1bxDjd1VZjzc1jAbi8JOmZ_AyoHBFgHpO&export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')" -O "emb_func_best.pth" && rm -rf /tmp/cookies.txt
```

## cd 回到 train_herbs_api路徑，當前路徑應該是train_herbs_api->code->pretrained所以要cd ..兩次!!!
```
cd ..
```

## 建立image，在terminal輸入
```
docker build -t herbs_train_image .
```

## 建立container，在terminal輸入
```
docker run -d --gpus all --shm-size 4G --name herbs_train_container -p 8008:8008 herbs_train_image
```

## 在網址欄中輸入就可以進到api網頁
host_url:8008/docs  
e.g. http://hc7.isl.lab.nycu.edu.tw:8008/docs

## 查看logs，在terminal輸入
```
docker logs herbs_train_container
```
