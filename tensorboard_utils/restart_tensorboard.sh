
ps | grep tensorboard 2> /dev/null
if [[ $? -eq 0 ]]; then
   ./stop_tensorboard.sh
fi

./start_tensorboard.sh

