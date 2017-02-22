
logdir="/media/mike/Main Storage/tensorflow-logs/mnist_test_logdir"

ps | grep tensorboard 2> /dev/null
if [[ $? -eq 0 ]]; then
   echo "Tensorboard is already running!"
   exit 1
else
   tensorboard --logdir="$logdir" 2> tensorboard.log &
   sleep 1
fi



