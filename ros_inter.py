import roslibpy
from time import sleep

# Constants/Definitions
host = "localhost"
port = 9090

start_instance_msg = "CONNECTED: "
message_refresh_wait = 1

gps_info = dict()
buffer = dict()
gps = '/GPS'


def ros_broadcast_connection(topic=gps, topic_dict=gps_info):
    """
    Manages the connection between the ROS server and broadcaster
    Additionally, it creates and broadcasts a topic for ROS listeners.

    :param topic: String Name of ROS topic preceded by /        e.g. /GPS
    :param topic_dict: Key/Value pairs to be broadcast
    :return:
    """

    # Initializes ROS connection manager
    broadcast_connection = roslibpy.Ros(host=host, port=port)

    # Prints bool value about ROS server connection status
    broadcast_connection.on_ready(lambda: (print(start_instance_msg), broadcast_connection.is_connected))

    # Runs without a timeout, terminates later
    broadcast_connection.run_forever()

    # Creates ROS Topic object for GPS information
    gps_topic = roslibpy.Topic(broadcast_connection, topic, 'std_msgs/String')

    while broadcast_connection.is_connected:
        # Publishes the GPS key value pairs to the GPS topic for all listeners
        gps_topic.publish(roslibpy.Message(topic_dict))

        print('SENDING')
        sleep(message_refresh_wait)

    # Cleans up instance when broadcast is over
    print('CONNECTION ENDED')
    gps_topic.unadvertise()
    broadcast_connection.terminate()


def ros_listen_connection(topic=gps):
    """
    Manages the connection between the ROS server and listener
    Additionally, it creates and subscribes to a topic for ROS listeners.

    :param topic: String Name of ROS topic preceded by /        e.g. /GPS
    :return:
    """
    # Initializes ROS connection manager
    listen_connection = roslibpy.Ros(host=host, port=port)

    # Runs with timeout, only requires single run to acquire data
    listen_connection.run()

    # Creates ROS Topic object for GPS information
    listener = roslibpy.Topic(listen_connection, topic, 'std_msgs/String')

    # Subscribes to ROS topic of GPS key value pairs
    listener.subscribe(lambda message: print(message['data']))

    listen_connection.terminate()
