# music_generator
Music generator from wav file with Keras stateful RNN

# development
We can develop this with docker container.

```
docker run --rm -p=6006:6006 -p=8888:8888 -v `pwd`/src:/srv -v `pwd`/data:/data -m 2g gw000/keras-full
```
