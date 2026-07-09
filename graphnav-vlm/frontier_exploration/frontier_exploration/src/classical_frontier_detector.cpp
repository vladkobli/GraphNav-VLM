/**
 * @file classical_frontier_detection.cpp
 * @author 
 * @brief Implementation of frontier-based exploration
 * @version 0.1
 * @date 2023-03-11
 * 
 * @copyright Copyright (c) 2023
 *
 */
#include "frontier_exploration/classical_frontier_detector.hpp"

#include <algorithm>

FrontierExplorer::FrontierExplorer()
: Node("frontier_explorer")
{
    // Get parameters
    region_size_thresh_ = this->declare_parameter("region_size_thresh", 12);
    preprocess_iterations_ = this->declare_parameter("preprocess_iterations", 0);
    marker_min_region_size_ = this->declare_parameter("marker_min_region_size", region_size_thresh_);
    marker_max_count_ = this->declare_parameter("marker_max_count", 35);
    robot_width_ = this->declare_parameter("robot_width", 0.5);
    marker_scale_ = this->declare_parameter("marker_scale", 0.08);
    occupancy_map_topic_ = this->declare_parameter("occupancy_map_msg", "map");

    // Subscribers/Publichers/Service setup
    map_subscription_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        occupancy_map_topic_, 1, std::bind(&FrontierExplorer::map_callback, this, _1));

    service_ = this->create_service<frontier_interfaces::srv::FrontierGoal>(
        "frontier_pose", std::bind(&FrontierExplorer::get_frontiers, this, _1, _2));

    marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("f_markers", 1);
    frontier_map_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("f_map", 1);

    //tf listner 
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

void FrontierExplorer::map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr recent_map)
{
    // auto width = recent_map->info.width;
    // auto height = recent_map->info.height;
    // RCLCPP_INFO(this->get_logger(),"Map recieved w: %d h: %d.", width, height);

    std::lock_guard<std::mutex> guard(mutex_);
    map_ = *recent_map;
}

void FrontierExplorer::get_frontiers(const std::shared_ptr<frontier_interfaces::srv::FrontierGoal::Request> request,
          std::shared_ptr<frontier_interfaces::srv::FrontierGoal::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "Received request, %d", request->goal_rank);

    // Copy the map
    std::unique_lock<std::mutex> lck(mutex_);
        nav_msgs::msg::OccupancyGrid map = map_;
    lck.unlock();

    // Pre-process the grid cell map
    std::vector<cell> processed = preprocessMap(map.data, map.info.width, map.info.height, preprocess_iterations_);

    // Compute frontier grid cell map
    frontierCellGrid_.clear();
	frontierCellGrid_ = computeFrontierCellGrid(processed, map.info.width, map.info.height);
	
	// Compute the Frontier Regions
	frontierRegions_.clear();
	frontierRegions_ = computeFrontierRegions(frontierCellGrid_, map.info.width, map.info.height, 
        map.info.resolution, map.info.origin.position.x, map.info.origin.position.y, region_size_thresh_);

    RCLCPP_INFO(
        this->get_logger(),
        "Detected %zu frontier region(s), min_region_size=%d, preprocess_iterations=%d.",
        frontierRegions_.size(),
        region_size_thresh_,
        preprocess_iterations_
    );

    nav_msgs::msg::OccupancyGrid f_map = map;

    // std::transform(frontierCellGrid_.begin(), frontierCellGrid_.end(), frontierCellGrid_.begin(),
    //            std::bind(std::multiplies<cell>(), std::placeholders::_1, 255));
    f_map.data = processed;
    frontier_map_publisher_->publish(f_map);

    publishFrontiers();

    // Get robot position
    geometry_msgs::msg::TransformStamped stransform;
    try {
        stransform = tf_buffer_->lookupTransform(
            odom_frame_,
            base_frame_,
            tf2::TimePointZero,
            tf2::durationFromSec(3)
        );
    }
    catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(this->get_logger(), "%s", ex.what());

        geometry_msgs::msg::PoseStamped goal_pose;
        goal_pose.header.stamp = this->get_clock()->now();
        goal_pose.header.frame_id = map_frame_;
        goal_pose.pose.orientation.w = 1.0;
        response->goal_pose = goal_pose;
        return;
    }
    
    // Create and init response message
    geometry_msgs::msg::PoseStamped goal_pose;
    goal_pose.header.stamp = this->get_clock()->now();
    goal_pose.header.frame_id = map_frame_;
    goal_pose.pose.orientation.w = 1.0;

    // No frontier found
    if (frontierRegions_.empty()) {
        RCLCPP_WARN(this->get_logger(), "No frontier regions available.");
        response->goal_pose = goal_pose;
        return;
    }

    // Requested rank is outside available frontier list
    if (request->goal_rank < 0 ||
        static_cast<size_t>(request->goal_rank) >= frontierRegions_.size()) {
        RCLCPP_WARN(
            this->get_logger(),
            "Requested frontier rank %d, but only %zu frontier(s) available.",
            request->goal_rank,
            frontierRegions_.size()
        );

        response->goal_pose = goal_pose;
        return;
    }

    // Find best goal based on position and size
    frontierRegion goal = selectFrontier(
        frontierRegions_,
        request->goal_rank,
        stransform.transform.translation.x,
        stransform.transform.translation.y
    );

    goal_pose.pose.position.x = goal.x;
    goal_pose.pose.position.y = goal.y;

    // Set the response
    response->goal_pose = goal_pose;

    RCLCPP_INFO(this->get_logger(), "Sending goal x: %f y: %f.",
        goal_pose.pose.position.x, goal_pose.pose.position.y);
}

void FrontierExplorer::publishFrontiers()
{
    visualization_msgs::msg::Marker::SharedPtr sphere_list(new visualization_msgs::msg::Marker);
    sphere_list->header.frame_id = map_frame_;
    sphere_list->header.stamp = this->get_clock()->now();
    sphere_list->type = visualization_msgs::msg::Marker::SPHERE_LIST;
    sphere_list->action = visualization_msgs::msg::Marker::ADD;
    sphere_list->scale.x = marker_scale_; // in meters
    sphere_list->scale.y = marker_scale_;
    sphere_list->scale.z = marker_scale_;
    // Set green and alpha(opacity)
    sphere_list->color.g = 1.0;
    sphere_list->color.a = 1.0;

    std::vector<frontierRegion> marker_regions;
    marker_regions.reserve(frontierRegions_.size());

    int min_marker_size = marker_min_region_size_;
    if (min_marker_size <= 0) {
        min_marker_size = region_size_thresh_;
    }

    for (const auto &reg : frontierRegions_) {
        if (reg.size >= min_marker_size) {
            marker_regions.push_back(reg);
        }
    }

    std::sort(
        marker_regions.begin(),
        marker_regions.end(),
        [](const frontierRegion &a, const frontierRegion &b) {
            return a.size > b.size;
        }
    );

    if (marker_max_count_ > 0 && marker_regions.size() > static_cast<size_t>(marker_max_count_)) {
        marker_regions.resize(static_cast<size_t>(marker_max_count_));
    }

    for(auto reg : marker_regions) {
        geometry_msgs::msg::Point p;
        p.x = reg.x;
        p.y = reg.y;
        p.z = 0.05;
        sphere_list->points.push_back(p);
    }
    marker_publisher_->publish(*sphere_list);
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    // Start processing data from the node as well as the callbacks and the timer
    rclcpp::spin(std::make_shared<FrontierExplorer>());
    
    // Shutdown the node when finished
    rclcpp::shutdown();
    return 0;
}
