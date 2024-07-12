#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <mrs_msgs/Vec4.h>  // Replace with the actual service header
#include <cmath>
#include <mutex>
#include <tf/tf.h>

// Global variable to store the previous best particle pose
geometry_msgs::Pose previous_pose;
bool is_first_message = true;  // Flag to check if this is the first message

// Mutex to synchronize access to shared resources
std::mutex mtx;

// Helper function to calculate the Euclidean distance between two poses
double calculateDistance(const geometry_msgs::Pose& pose1, const geometry_msgs::Pose& pose2)
{
    return sqrt(pow(pose1.position.x - pose2.position.x, 2) +
                pow(pose1.position.y - pose2.position.y, 2) +
                pow(pose1.position.z - pose2.position.z, 2));
}

// Callback function for /best_particle_pose
void bestParticlePoseCallback(const geometry_msgs::Pose::ConstPtr& msg, ros::ServiceClient& client, double distance_threshold)
{

    std::lock_guard<std::mutex> lock(mtx); // Lock the mutex to synchronize access

    // Calculate the distance from the previous pose
    if (!is_first_message)
    {
        double distance = calculateDistance(previous_pose, *msg);

        // Check if the distance exceeds the threshold
        if (distance < distance_threshold)
        {
            // ROS_INFO("New pose is too close to the previous one. Skipping service call.");
            return;
        }
    }
    else
    {
        is_first_message = false;
    }

    // Update the previous pose
    previous_pose = *msg;

    // Create the service request and response objects
    mrs_msgs::Vec4 srv;
    srv.request.goal[0] = msg->position.x;
    srv.request.goal[1] = msg->position.y;
    srv.request.goal[2] = msg->position.z;
    // Get yaw from orientation quaternion
    tf::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    srv.request.goal[3] = yaw;
    ROS_INFO("Sending goal: x=%f, y=%f, z=%f, yaw=%f", srv.request.goal[0], srv.request.goal[1], srv.request.goal[2], srv.request.goal[3]);
    // srv.request.goal[3] = 1.0;  // Example value, adjust as needed
    
    client.waitForExistence();
    // Call the service
    if (client.call(srv))
    {
        if (srv.response.success)
        {
            ROS_INFO("Service call successful: %s", srv.response.message.c_str());
        }
        else
        {
            ROS_ERROR("Service call was not successful: %s", srv.response.message.c_str());
        }
    }
    else
    {
        ROS_ERROR("Failed to call service Vec4Service. Ensure the service server is running and check the service implementation.");
    }
}

int main(int argc, char **argv)
{
    // Initialize the ROS node
    ros::init(argc, argv, "vec4_service_client");
    ros::NodeHandle nh("~");

    // Define the distance threshold
    double distance_threshold;
    std::string service_name, goal_topic;
    nh.param("distance_threshold", distance_threshold, 0.5);  // Default threshold of 0.5 meters
    nh.param("goal_service", service_name, std::string("/vec4_service"));
    nh.param("goal_topic", goal_topic, std::string("/best_particle_pose"));
    ROS_WARN("Goal topic: %s", goal_topic.c_str());
    ROS_WARN("Goal service: %s", service_name.c_str());

    // Create a service client for Vec4Service
    ros::ServiceClient client = nh.serviceClient<mrs_msgs::Vec4>(service_name);

    // Subscribe to the /best_particle_pose topic
    ros::Subscriber sub = nh.subscribe<geometry_msgs::Pose>(goal_topic, 10,
        boost::bind(bestParticlePoseCallback, _1, boost::ref(client), distance_threshold));

    // Spin to keep the node running
    ros::spin();

    return 0;
}
