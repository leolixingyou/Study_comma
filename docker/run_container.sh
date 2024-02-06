sudo docker run -it \
-v "$(pwd)/../../":/workspace \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-e XDG_RUNTIME_DIR=/tmp/runtime-root \
-e DISPLAY=unix$DISPLAY \
--net=host \
--gpus all \
--privileged \
--name study_comma \
study_comma_image


