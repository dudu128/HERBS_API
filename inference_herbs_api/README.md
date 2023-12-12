1. ### 建立image
docker build -t herbs_inference_image .

2. ### 建立container
docker run -idt --gpus all --shm-size 8G --name herbs_inference_container -p 8001:8001 herbs_inference_image

3. ### 查看logs
docker logs  herbs_inference_container