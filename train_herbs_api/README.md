1. ### 建立image
docker build -t herbs_train_image .

2. ### 建立container
docker run -idt --gpus all --shm-size 8G --name herbs_train_container -p 8008:8008 herbs_train_image

3. ### 查看logs
docker logs  herbs_train_container