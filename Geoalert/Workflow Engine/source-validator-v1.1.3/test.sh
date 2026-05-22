for FILE in ./tests/*.json;
do sleep 1 && echo Sending request: $FILE \
&& docker compose exec queue rabbitmqadmin -u user -p password publish \
  exchange=amq.default routing_key=validate-source.tasks.queue \
  properties='{"content_type":"application/json"}' \
  payload="`cat $FILE`" \
  payload_encoding='string';
done
