## 建立image，在terminal輸入
docker build -t herbs_train_image .

## 建立container，在terminal輸入
docker run -idt --gpus all --shm-size 8G --name herbs_train_container -p 8008:8008 herbs_train_image

## 在網址欄中輸入就可以進到api網頁
server:8008/docs  
e.g. http://hc7.isl.lab.nycu.edu.tw:8008/docs

## 查看logs，在terminal輸入
docker logs  herbs_train_container
