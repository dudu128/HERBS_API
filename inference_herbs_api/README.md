## 建立image，在terminal輸入
```
docker build -t herbs_inference_image .
```

## 建立container，在terminal輸入
```
docker run -idt --gpus all --shm-size 8G --name herbs_inference_container -p 8001:8001 herbs_inference_image
```

## 在網址欄中輸入就可以進到api網頁
host_url:8001/docs  
e.g. http://hc7.isl.lab.nycu.edu.tw:8001/docs

## 查看logs，在terminal輸入
```
docker logs herbs_inference_container
```
