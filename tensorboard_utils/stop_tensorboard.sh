
#ps | grep tensorboard 2> /dev/null
ps | grep tensorboard 2> /dev/null
if [[ $? -eq 0 ]]; then
   echo "Stopping Tensorboard"
   killall tensorboard
else
   echo "Tensorboard isn't running!"
   exit 1
fi

