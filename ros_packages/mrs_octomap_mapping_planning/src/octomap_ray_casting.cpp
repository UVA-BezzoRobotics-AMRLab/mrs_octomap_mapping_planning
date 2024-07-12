#include <iostream>
#include <vector>
#include <ros/ros.h>
#include <octomap_msgs/Octomap.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <geometry_msgs/Pose.h>
#include <tf/tf.h>
#include <random>
#include <cmath>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <thread>
#include <mutex>
#include <algorithm>
#include <Eigen/Dense>
#include <nav_msgs/Odometry.h>

using Array3D = std::vector<std::vector<double>>;

using namespace Eigen;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


// Global variables
octomap::OcTree* receivedOctomap = nullptr;
std::vector<double> hitScore;
std::vector<octomap::point3d> hitCoords;
octomap::point3d currentPosition;
octomap::point3d currentOrient;
Array3D target;

int ytargetSize, ztargetSize, rows, cols;
float leftY, xTarget, targetRes;
double fov, rayLength, angle_offset, increment, thetaIncrement, phiIncrement;
// Create a random number generator engine
std::default_random_engine gen;
// Mutex for locking shared resources during the update process
std::mutex mtx;
MatrixXd initX_train; // Replace this with your actual training data
VectorXd initY_train; // Replace this with your actual target values
// int ytargetSize = 12;  // How wide is the target?
// int ztargetSize = 6;  // How tall is the target?
// float leftY = 7.0; // Where does the target start in the y
// float xTarget = 8.5; //where does the target sit in the static x-plane
// float targetRes = 1; // What is the resolution of the target?
// int rows = ytargetSize*ztargetSize; // Total number coordinates
// int cols = 3; // Replace this with the desired number of columns

double wrapToPi(double angle) {
    angle = std::fmod(angle + M_PI, 2 * M_PI);
    if (angle < 0)
        angle += 2 * M_PI;
    
    if (angle >= M_PI)
        angle -= 2 * M_PI;
    
    return angle;
}

void makeTarget(){
  // Initialize an empty MatrixXd with zeros
  initX_train = MatrixXd::Zero(rows, cols);
  float leftZ = 0.0;
  float yTarget, zTarget;
  int count = 0;
  for (int i = 0; i<ytargetSize; ++i){
      yTarget = leftY - i*targetRes;
      for (int j = 0; j<ztargetSize; ++j){
      zTarget = leftZ + j*targetRes;
      target.push_back({xTarget,yTarget,zTarget});
      initX_train(count,0) = xTarget; 
      initX_train(count,1) = yTarget;
      initX_train(count,2) = zTarget;
      count+=1;
      }
  }
}

std::array<float,3> targetCenter;

// Define rays
int numRays = 5;  // Number of rays to generate
// double fov = M_PI/2;  // Field of view of the sensor
// double increment = fov;
// double rayLength = 6.0; // Length of the ray
// Particles
const int numParticles = 100;
const int bounds = 10;
// Create a distribution for the range of particle positions you want
std::uniform_real_distribution<double> distv(0, 2);

struct particle {
  // Define the fields of your structure
  octomap::point3d position;
  octomap::point3d orient;
  double bestScore = -1e4;
  octomap::point3d bestPosition;
  double bestYaw = 0;
  std::array<double,3> vel = {0,0,0};
  // Create the NLOpt optimization problem with L-BFGS algorithm
   // Two hyperparameters (theta, sigma)
};

struct gpStruct {
  MatrixXd X_train;
  VectorXd Y_train;
  MatrixXd X_train_data = Eigen::MatrixXd::Zero(1000,3);
  VectorXd Y_train_data = Eigen::VectorXd::Zero(1000);
  int num_data = 0;
  VectorXd mu;
  double bestScore;
};

particle particleArr[numParticles]; // Declare particle array
particle globalBest;
gpStruct gp;
// PSO parameters
double w = 0.8; 
const double c1 = 0.1; const double c2 = 0.1; 


// Initialize the gaussian process for the vehicle
void initializeGP(){
  gp.X_train = initX_train;
  gp.Y_train = VectorXd::Zero(rows);
  gp.mu = VectorXd::Zero(rows);
}

// Initialize parameters of the scoring functions
double noise = 1;
float A,B,C,D;
MatrixXd transformedPoints;
VectorXd translationVector;
Matrix3d rotationMatrix;
VectorXd centroid;

// -------------------------------------------------------------

// Function to compute the hyperplane equation coefficients
void hyperplaneEquation(const MatrixXd& points, float& A, float& B, float& C, float& D, VectorXd& centroid) {
    // Calculate the centroid of the points
    centroid = points.colwise().mean();
    // Calculate the covariance matrix and perform eigendecomposition
    MatrixXd centered = points.rowwise() - centroid.transpose();
    JacobiSVD<MatrixXd> svd(centered, ComputeThinU | ComputeThinV);
    VectorXd normal = svd.matrixV().col(points.cols() - 1);
    // Calculate the constant term D
    D = -normal.dot(centroid);
    // Normalize the normal vector
    normal = normal/normal.norm();

    // Extract components for the equation Ax + By + Cz + D = 0
    A = normal(0);
    B = normal(1);
    C = normal(2);
}
// -------------------------------------------------------------


// // Function to transform points to a new hyperplane
void transformToNewHyperplane(const MatrixXd points, float A, float B, float C, float D,
                              MatrixXd& transformedPoints, VectorXd& translationVector,
                              Matrix3d& rotationMatrix, VectorXd& centroid) {
    // Calculate the centroid of the points
    centroid = points.colwise().mean();

    // Translate the points to bring the centroid to the origin
    MatrixXd translatedPoints = points.rowwise() - centroid.transpose();

    // Calculate the normal vector of the new hyperplane
    Vector3d newNormal(A, B, C);
    newNormal.normalize();
    // std::cout << "newNormal: " << newNormal << std::endl;
    // Calculate the rotation matrix to align the new normal with (0, 0, 1)
    Vector3d rotationAxis = newNormal.cross(Vector3d(0, 0, 1));
    double rotationAngle = std::acos(newNormal.dot(Vector3d(0, 0, 1)) / (newNormal.norm() * Vector3d(0, 0, 1).norm()));
    if (rotationAxis.isZero(1e-6)) {
        rotationMatrix = Matrix3d::Identity();
    } else {
        rotationMatrix = AngleAxisd(rotationAngle, rotationAxis).toRotationMatrix();
    }

    // // Transform the points using the rotation
    MatrixXd rotatedPoints = (rotationMatrix * translatedPoints.transpose()).transpose();

    // // Translate the points back to the new hyperplane
    translationVector = centroid - (centroid.dot(newNormal) + D) * newNormal;
    transformedPoints = rotatedPoints.rowwise() + translationVector.transpose();
}

// Function to transform points to a new point based on the given translation and rotation
Vector3d transformToNewPoint(const Vector3d& points, VectorXd& translationVector,
                         Matrix3d& rotationMatrix, VectorXd& centroid) {
    // Translate the points to bring the centroid to the origin
    MatrixXd translatedPoints = points.transpose() - centroid.transpose();

    // Transform the points using the rotation
    MatrixXd rotatedPoints = (rotationMatrix * translatedPoints.transpose()).transpose();

    // Translate the points back to the new hyperplane
    Vector3d transformedPoints = rotatedPoints.transpose() + translationVector;

    return transformedPoints;
}
// -------------------------------------------------------------

// Randomize particles
octomap::point3d randomizeParticle(std::array<float,3> loc){
    // Generate a random position
    loc = {0,0,0};
  std::uniform_real_distribution<double> distx(loc[0]-bounds, loc[0]+bounds);
  std::uniform_real_distribution<double> disty(loc[1]-bounds, loc[1]+bounds);
  std::uniform_real_distribution<double> distz(1, loc[2]+bounds);

  double xVal = distx(gen);
  double yVal = disty(gen);
  double zVal = distz(gen);
  return octomap::point3d(xVal,yVal,zVal);
}


// Update best particle
void updateBest(particle p){
    if (p.bestScore > globalBest.bestScore){
        globalBest.bestScore = p.bestScore;
        globalBest.bestYaw = p.bestYaw;
        globalBest.bestPosition = p.bestPosition;
    }
}

// Function to calculate the center of the target
void calculateCenter(){
  int colCount = target[0].size();
  int rowCount = target.size();
  for(int j = 0; j<colCount; ++j){
    double sum = 0.0;
    for(int i = 0; i<rowCount; ++i){
      sum += target[i][j];
    }
    targetCenter[j] = sum / rowCount;
  }
}
// -------------------------------------------------------------

// update the target
void updateTarget(){
  // Initialize an empty MatrixXd with zeros
  target.clear();
  float leftZ = 0.3;
  float yTarget, zTarget;
  int count = 0;
  for (int i = 0; i<ytargetSize; ++i){
    yTarget = leftY - i*targetRes;
    for (int j = 0; j<ztargetSize; ++j){
      zTarget = leftZ + j*targetRes;
      if (gp.mu(i*j)<0.01){
        target.push_back({xTarget,yTarget,zTarget});
        count+=1;
      }
    }
  }
  calculateCenter();
}

// Function for orientation to center of the target
std::array<double,2> calculateOrientation(octomap::point3d pos){
  // Returns (theta_d,phi_d)
  std::array<double,2> orient;
  double diffx = targetCenter[0]-pos.x();
  double diffy = targetCenter[1]-pos.y();
  double diffz = targetCenter[2]-pos.z();
  double theta_d = atan2(diffz,sqrt(pow(diffx,2)+pow(diffy,2)));
  double phi_d = atan2(diffy,diffx);
  orient[0] = theta_d + 3*M_PI/2; //orient[1] = phi_d;
  orient[1] = phi_d + M_PI; 
  return orient; 
}
// -------------------------------------------------------------

// Callback function for the position of the firefly
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg){

  // Extract the coordinates from the Pose message
  double x = msg->pose.pose.position.x;
  double y = msg->pose.pose.position.y;
  double z = msg->pose.pose.position.z;

 // Extract the orientation from the Pose message (in quaternion format)
  double qx = msg->pose.pose.orientation.x;
  double qy = msg->pose.pose.orientation.y;
  double qz = msg->pose.pose.orientation.z;
  double qw = msg->pose.pose.orientation.w;

  // Convert the quaternion to roll, pitch, and yaw
  double roll, pitch, yaw;
  tf::Matrix3x3(tf::Quaternion(qx, qy, qz, qw)).getRPY(roll, pitch, yaw);

  // Store the position and orientation as octomap::point3d objects
  currentPosition = octomap::point3d(x, y, z);
  currentOrient = octomap::point3d(roll, pitch, yaw);
}
// -------------------------------------------------------------

// Callback function for the octomap message
void octomapCallback(const octomap_msgs::Octomap::ConstPtr& msg)
{

  // Convert the octomap message to an octomap object
  if (receivedOctomap)
    delete receivedOctomap;

  receivedOctomap = dynamic_cast<octomap::OcTree*>(octomap_msgs::fullMsgToMap(*msg));
}
// -------------------------------------------------------------

// Kernel function for GPR
MatrixXd kernel(const MatrixXd& X1, const MatrixXd& X2, double length_scale, double noise_level) {
    int m = X1.rows();
    int n = X1.cols();
    int p = X2.rows();

    MatrixXd K(m, p);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sqdist = 0.0;
            for (int k = 0; k < n; k++) {
                sqdist += (X1(i, k) - X2(j, k)) * (X1(i, k) - X2(j, k));
            }
            K(i, j) = noise_level*noise_level * exp(-0.5 / (length_scale) * sqdist);
        }
    }

    // for (int i = 0; i < m; i++) {
    //     K(i, i) += noise * noise;
    // }

    return K;
}
// -------------------------------------------------------------

// Objective function for PSO
double objectiveFunction(const VectorXd x, void *data){
    // Cast the void* data to the structure type
    double nll = 0.0; 
    double noise_level = x[0];
    double length_scale = x[1]; 

    gpStruct* gp_struct = static_cast<gpStruct*>(data);
    MatrixXd X_train = gp_struct->X_train;
    VectorXd y_train = gp_struct->Y_train;
    
    int numRows = X_train.rows();
    int numCols = X_train.cols();

    MatrixXd K = kernel(X_train, X_train, length_scale, noise_level);
    // Calculate Cholesky decomposition of K
    LLT<MatrixXd> lltOfK(K);
    // std::cout<<lltOfK.info()<<std::endl;
    // Check if Cholesky decomposition is valid
    if (lltOfK.info() != Success) {
        // Handle the case when Cholesky decomposition fails (matrix is not positive definite)
        return 1e7;
    }

    // Get the lower triangular matrix L from the Cholesky factorization
    MatrixXd L = lltOfK.matrixL();
    // std::cout << "L" << std::endl;
    // Solve L * S1 = Y_train
    VectorXd S1 = L.triangularView<Lower>().solve(y_train);
    // std::cout << "S1" << std::endl;
    // Solve L^T * S2 = S1
    VectorXd S2 = L.transpose().triangularView<Upper>().solve(S1);
    // std::cout << "S2" << std::endl;

    // Compute the log likelihood
    double log_likelihood = 0.5 * numRows * log(2 * M_PI) + 0.5 * y_train.dot(S2) + L.diagonal().array().log().sum();
    // std::cout << "log_likelihood: " << log_likelihood << std::endl;
    // std::cout << "log_likelihood: " << log_likelihood << std::endl;
    return log_likelihood;
};
// -------------------------------------------------------------

// Particle Swarm Optimization function
std::vector<double> PSO(int num_particles, int max_iterations, gpStruct gp_data) {
    const int num_hyperparameters = 2; // Number of hyperparameters (length scale and noise level)
    const double inertia_weight = 0.9;
    const double cognitive_weight = 2.0;
    const double social_weight = 2.0;
    const double velocity_limit = 0.1;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    // Initialize particles with random positions and velocities
    MatrixXd particles(num_particles, num_hyperparameters);
    MatrixXd velocities(num_particles, num_hyperparameters);
    VectorXd best_particle(num_hyperparameters);
    double best_fitness = -std::numeric_limits<double>::max();

    for (int i = 0; i < num_particles; i++) {
        for (int j = 0; j < num_hyperparameters; j++) {
            particles(i, j) = distribution(generator);
            velocities(i, j) = distribution(generator);
        }

        // Evaluate the objective function for each particle
        double fitness = objectiveFunction(particles.row(i), &gp_data);

        if (fitness > best_fitness) {
            best_fitness = fitness;
            best_particle = particles.row(i);
        }
    }

    // PSO iterations
    for (int iter = 0; iter < max_iterations; iter++) {
        for (int i = 0; i < num_particles; i++) {
            // Update particle velocity
            for (int j = 0; j < num_hyperparameters; j++) {
                double r1 = distribution(generator);
                double r2 = distribution(generator);
                velocities(i, j) = inertia_weight * velocities(i, j) +
                                  cognitive_weight * r1 * (best_particle(j) - particles(i, j)) +
                                  social_weight * r2 * (best_particle(j) - particles(i, j));

                // Apply velocity limits
                if (velocities(i, j) > velocity_limit) {
                    velocities(i, j) = velocity_limit;
                }
                if (velocities(i, j) < -velocity_limit) {
                    velocities(i, j) = -velocity_limit;
                }
            }

            // Update particle position
            particles.row(i) += velocities.row(i);

            // Evaluate the objective function for the updated particle
            double fitness = objectiveFunction(particles.row(i), &gp_data);

            // Update the best position and fitness for the particle
            if (fitness > best_fitness) {
                best_fitness = fitness;
                best_particle = particles.row(i);
            }
        }
    }
    // Return the best hyperparameters
    std::vector<double> params(num_hyperparameters);
    for (int j = 0; j < num_hyperparameters; j++) {
        params[j] = best_particle(j);
    }
    return params;
}
// -------------------------------------------------------------

// Function to avoid overtraining data
void clean_data(MatrixXd& X, VectorXd& y, int& num_data){
    int count = 0; 
    MatrixXd X_sub = Eigen::MatrixXd::Zero(X.rows(),3);
    VectorXd y_sub = Eigen::VectorXd::Zero(y.rows());
    VectorXd include = VectorXd::Ones(y.rows());
    X_sub.row(0) = X.row(0); y_sub(0) = y(0);
    for (int i = 0; i < num_data-1; i++){
        for (int j = i+1; j < num_data; j++){
            if (include(j) <1 || include(i) <1){
                continue;
            }
            Eigen::Vector3d point1 = X.row(i);
            // std::cout << "Point 1: " << point1 << std::endl;
            Eigen::Vector3d point2 = X.row(j);
            // std::cout << "Point 2: " << point2 << std::endl;
            //Calc euclidean distance
            double dist = (point2-point1).norm();
            if (dist<3){
                include(j) = 0;
            }
        }
    }
    // Remove data points that are too close to each other
    for (int i = 0; i<num_data; i++){
        if (include(i) > 0){
            X_sub.row(count) = X.row(i);
            y_sub(count) = y(i);
            count++;
        }
    }
    X = X_sub;
    y = y_sub;
    num_data = count;
}
 
void predict(const Eigen::MatrixXd X_new, Eigen::MatrixXd X_train, Eigen::VectorXd y_train, Eigen::VectorXd& mean, std::vector<double> params) {
    double noise_level = params[0];
    double length_scale = params[1]; 
    int n = X_train.rows();
    int m = X_new.rows();

    Eigen::MatrixXd K(n, n);
    Eigen::MatrixXd K_star(m, n);
    Eigen::MatrixXd K_star_star(m, m);

    K_star = kernel(X_new, X_train,length_scale,noise_level);

    K = kernel(X_train, X_train,length_scale,noise_level);
    K = K + Eigen::MatrixXd::Identity(n,n)*noise_level*noise_level;
    K_star_star = kernel(X_new, X_new,length_scale,noise_level);
    K_star_star = K_star_star + Eigen::MatrixXd::Identity(m,m)*noise_level*noise_level;
    Eigen::MatrixXd K_inv = K.inverse();
    Eigen::MatrixXd K_star_inv = K_star * K_inv;
    
    mean =  K_star_inv* y_train;
    // variance = K_star_star - K_star_inv * K_star.transpose(); // Transpose K_inv_K_star for correct dimension
}
// -------------------------------------------------------------

// Function to find if an array is equal to any arrays in a target vector
bool updateTrainData(const octomap::point3d& point, const Array3D& arrayVector, MatrixXd& x_train_data, VectorXd& y_train_data, int& num_data, int& whichPoint) {
  // ROS_INFO("Im here!");
  if (arrayVector.empty()){
    // ROS_INFO("IMONLYHERE");
    return false;
  }
  int count = 0;
  for (const auto& array : arrayVector) {
    //  ROS_INFO("(%f,%f,%f) >> (%f,%f,%f)",array[0],array[1],array[2],point.x(),point.y(),point.z());
    // ROS_INFO("%f,%f,%s,%s",array[0],point.x(),typeid(array[0]).name(),typeid(point.x()).name());
    // if (array[0]==point.x() && array[1]==point.y()){
    //   ROS_INFO("(%f,%f,%f) >> (%f,%f,%f)",array[0],array[1],array[2],point.x(),point.y(),point.z());
    // }
    float targetDist = pow(array[0]-point.x(),2) + pow(array[1]-point.y(),2) + pow(array[2]-point.z(),2);
    if (targetDist<targetRes) {
      x_train_data(num_data,0) = point.x(); x_train_data(num_data,1) = point.y(); x_train_data(num_data,2) = point.z();
      y_train_data(num_data) = 1;
      whichPoint = count;
      num_data++;
      // ROS_INFO("(%f,%f,%f) >> (%f,%f,%f) >> Dist: %f",array[0],array[1],array[2],point.x(),point.y(),point.z(),targetDist);
      return true;
    }
    count+=1;
  }
  return false;
}
// -------------------------------------------------------------

// Function to calculate score of a particle
void particleScore(particle& p){
  octomap::point3d direction; //  Define the direction of the ray
  double voxelSize = receivedOctomap->getResolution(); // Voxel size of the OctoMap
  int numSteps = static_cast<int>(rayLength / voxelSize); // Number of steps along the ray
//   ROS_INFO("%i,%f,%f",numSteps,voxelSize,rayLength);
  Array3D pHitCoords;
  std::array<double,2> orient;
  orient = calculateOrientation(p.position); // Calculate orientation correction to face center of target
  gpStruct gp_particle = gp;
  

  // ROS_INFO("orient at (%f,%f); particle at (%f,%f,%f); target center at (%f,%f,%f)",orient[0],orient[1],p.position.x(),p.position.y(),p.position.z(),
  //                                                                       targetCenter[0],targetCenter[1],targetCenter[2]);
  double particleScore;
  octomap::OcTreeKey key;

  double phi_min = M_PI; double phi_max = -M_PI;
  Vector3d phi_lmin = {1e3,1e3,1e3}; Vector3d phi_lmax = {-1e3,-1e3,-1e3};
  double theta_min = M_PI; double theta_max = -M_PI;
  Vector3d theta_lmin = {1e3,1e3,1e3}; Vector3d theta_lmax = {-1e3,-1e3,-1e3};

  for (int i = 0; i < numRays + 1; ++i) {
    double phi_sub = i * phiIncrement - increment / 2;
    double phi = orient[1] + phi_sub;
    for (int j = 0; j < numRays + 1; ++j) {
      double theta_sub = j * thetaIncrement - increment / 2;
      double thetaZ = theta_sub + 3*M_PI/2;
      double theta = orient[0] + theta_sub;
      float x = std::sin(theta) * std::cos(phi);
      float y = std::sin(theta) * std::sin(phi);
      float z = std::cos(thetaZ); // No pitch in camera ---->std::cos(theta);
      direction = octomap::point3d(x,y,z);
      // std::cout << "direction: " << direction << std::endl; 
      for (int i = 0; i < numSteps; ++i) {
        octomap::point3d currentPoint; //= position + (i * voxelSize * direction);
        currentPoint.x() = p.position.x() + direction.x() * i * voxelSize;
        currentPoint.y() = p.position.y() + direction.y() * i * voxelSize;
        currentPoint.z() = p.position.z() + direction.z() * i * voxelSize;
        // std::cout << "currentPoint: " << currentPoint << std::endl;
        // receivedOctomap->coordToKeyChecked(currentPoint, key); // Find closest voxel to position
        // octomap::point3d closestVoxel = receivedOctomap->keyToCoord(key);
        // if (i==0 && j == 0){
        // ROS_INFO("Position:(%f,%f,%f); Direction:(%f,%f,%f); Phi: %f; Theta: %f; Orient: (%f,%f); Center: (%f,%f,%f)",p.position.x(),p.position.y(),p.position.z(),
        //           direction.x(),direction.y(),direction.z(),phi,theta,orient[0],orient[1],targetCenter[0],targetCenter[1],targetCenter[2]);
        // }
        // If position == target break
        int whichTarget;
        bool ifTarget = updateTrainData(currentPoint,target,gp_particle.X_train_data,gp_particle.Y_train_data,gp_particle.num_data,whichTarget);
        if (ifTarget && gp.mu(whichTarget)<0.5){
          //find distance between p.position and currentPoint
          Vector3d pt, hitPt;
          pt = {currentPoint.x(),currentPoint.y(),currentPoint.z()};
          hitPt = transformToNewPoint(pt,translationVector,
                              rotationMatrix,centroid);

          if (phi_sub < phi_min) {phi_min = phi_sub;}
          if (phi_sub > phi_max) {phi_max = phi_sub;}
          if (theta_sub < theta_min) {theta_min = theta_sub;}
          if (theta_sub > theta_max) {theta_max = theta_sub;}
          if (hitPt(1) < phi_lmin(1)) {phi_lmin = hitPt;}
          if (hitPt(1) > phi_lmax(1)) {phi_lmax = hitPt;}
          if (hitPt(0) < theta_lmin(0)) {theta_lmin = hitPt;} 
          if (hitPt(0) > theta_lmax(0)) {theta_lmax = hitPt;}

          break;
        }
        // octomap::OcTreeNode* node = receivedOctomap->search(currentPoint); // Retrieve the OcTreeNode for the current point
        bool node = false;
        
        if (node || i == numSteps-1) {
          gp_particle.X_train_data(gp_particle.num_data,0) = currentPoint.x(); 
          gp_particle.X_train_data(gp_particle.num_data,1) = currentPoint.y();
          gp_particle.X_train_data(gp_particle.num_data,2) = currentPoint.z();
          gp_particle.Y_train_data(gp_particle.num_data) = 0;
          gp_particle.num_data++;
          break;
        }
      }
    }
  }
  
  std::vector<double> params(2);
//   params = PSO(10,10,gp_particle);
    params = {1,1};
  // std::cout << "made it here: " << params[0] << " " << params[1] << std::endl;
  // ROS_INFO("made it here: (%f)", gp_particle.Y_train.sum());
//   std::cout << "made it here: (" << gp_particle.Y_train.sum() << ")" << std::endl;

  
  if (gp_particle.num_data>1 && gp_particle.Y_train_data.sum()>1){
    //clean data and then predict (minimizes overfitting)
    clean_data(gp_particle.X_train_data,gp_particle.Y_train_data,gp_particle.num_data);

    MatrixXd X_train_data = gp_particle.X_train_data.block(0,0,gp_particle.num_data,3);
    VectorXd y_train_data = gp_particle.Y_train_data.head(gp_particle.num_data);
    // std::cout << "num_data: " << gp_particle.num_data << std::endl;
    predict(gp_particle.X_train, X_train_data, y_train_data, gp_particle.mu, params);
  
    Vector3d rbtPt,tranRbtPt;
    rbtPt = {p.position.x(),p.position.y(),p.position.z()};
    tranRbtPt = transformToNewPoint(rbtPt,translationVector,
                                    rotationMatrix,centroid);

    // Find gamma values
    double gamma_s;
    double gamma_d_sub = abs(phi_max-phi_min)/fov * abs(theta_max-theta_min)/fov;
    double gamma_d = 1/(1+exp(20*(gamma_d_sub-0.75)));
    double gamma_s_v1 = (tranRbtPt-phi_lmin).norm();
    double gamma_s_v2 = (tranRbtPt-phi_lmax).norm();
    double gamma_s_v = (phi_lmax - phi_lmin).norm();
    double gamma_s_h1 = (tranRbtPt-theta_lmin).norm();
    double gamma_s_h2 = (tranRbtPt-theta_lmax).norm();
    double gamma_s_h = (theta_lmax - theta_lmin).norm();
    if (gamma_s_v == 0 || gamma_s_h == 0){
      gamma_s = 0;
    }
    else{
      double gamma_s_sub = abs(gamma_s_v1-gamma_s_v2)/gamma_s_v * abs(gamma_s_h1-gamma_s_h2)/gamma_s_h;
      gamma_s = 1/(1+exp(20*(gamma_s_sub-0.75)));
    }
    // gamma_s = 1; gamma_d = 1;
    particleScore = gamma_s*gamma_d*(gp_particle.mu.sum() - gp.mu.sum());
    //print out gamma_s_subTheta and gamma_s_subPhi and all their variables
    // std::cout << "gamma_s: " << gamma_s << " gamma_d: " << gamma_d << " particleScore: " << particleScore << std::endl;
  }
  else{
    particleScore = 0;
  }
  // std::cout << "gp_mu: " << gp.mu.sum() << " gp_particle_mu: " << gp_particle.mu.sum() << std::endl;
  // std::cout << "phi_min: " << phi_min << " phi_max: " << phi_max << " theta_min: " << theta_min << " theta_max: " << theta_max << std::endl; 
  // ROS_INFO("particleScore: %f", particleScore);
  
  if (particleScore>p.bestScore){
    p.bestScore = particleScore;
    p.bestPosition = p.position;
    p.bestYaw = orient[1] + angle_offset;
  }
}
// -------------------------------------------------------------

// Function to find which particle is the best target
void updateParticles(particle globalBest, particle& p){
  particleScore(p);
  // ROS_INFO("BestScore:%f; Best Position:(%f,%f,%f)",p.bestScore,p.bestPosition.x(),p.bestPosition.y(),p.bestPosition.z());
  // Check if particle should be rerandomized

  double xDiff = pow(p.position.x() - globalBest.bestPosition.x(),2);
  double yDiff = pow(p.position.y() - globalBest.bestPosition.y(),2);
  double zDiff = pow(p.position.z() - globalBest.bestPosition.z(),2);
  if (xDiff + yDiff + zDiff < 0.01){
    p.position = randomizeParticle(targetCenter);
    p.bestScore = -1e4;
    p.bestPosition = {0,0,0};
    p.bestYaw = 0;
  }
  // If there is a change, update all the particles best positions and move according to PSO algorithm
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0,1);
  double r1 = dist(gen); double r2 = dist(gen);
  double velX = w*p.vel[0] + c1*r1*(p.bestPosition.x()-p.position.x()) +
                    c2*r2*(globalBest.bestPosition.x()-p.position.x());
  double velY = w*p.vel[1] + c1*r1*(p.bestPosition.y()-p.position.y()) +
                    c2*r2*(globalBest.bestPosition.y()-p.position.y());
  double velZ = w*p.vel[2] + c1*r1*(p.bestPosition.z()-p.position.z()) +
                    c2*r2*(globalBest.bestPosition.z()-p.position.z());
  p.vel = {velX,velY,velZ};
  // ROS_INFO("Velocities: (%f,%f,%f)",velX,velY,velZ);
  p.position.x() += velX;
  p.position.y() += velY;
  p.position.z() += velZ;
}
// -------------------------------------------------------------

// Function to initialize particles for PSO
void initializeParticles(){
  octomap::point3d pos;
  for(int i = 0; i<numParticles; ++i){
    pos = randomizeParticle(targetCenter);
    particleArr[i].position = pos;
    particleArr[i].bestPosition = pos;
    particleScore(particleArr[i]);
    particleArr[i].vel = {distv(gen),distv(gen),distv(gen)};
    updateBest(particleArr[i]);
  }
}
// -------------------------------------------------------------

// Function to calculate score of a particle
void updateVehicleGP(gpStruct& gp){
  octomap::point3d direction; //  Define the direction of the ray
  double voxelSize = receivedOctomap->getResolution(); // Voxel size of the OctoMap
  int numSteps = static_cast<int>(rayLength / voxelSize); // Number of steps along the ray
//   ROS_INFO("%i,%f,%f",numSteps,voxelSize,rayLength);
  Array3D pHitCoords;
  double yaw = wrapToPi(currentOrient.z());
  // std::cout << "yaw: " << yaw << std::endl;
  
  // ROS_INFO("orient at (%f,%f); particle at (%f,%f,%f); target center at (%f,%f,%f)",orient[0],orient[1],p.position.x(),p.position.y(),p.position.z(),
  //                                                                       targetCenter[0],targetCenter[1],targetCenter[2]);
  double particleScore;
  octomap::OcTreeKey key;

  for (int i = 0; i < numRays + 1; ++i) {
    double phi_sub = i * phiIncrement - increment / 2;
    double phi = yaw + phi_sub;
    for (int j = 0; j < numRays + 1; ++j) {
      double theta_sub = j * thetaIncrement - increment / 2;
      double thetaZ = theta_sub + 3*M_PI/2;
      double theta = theta_sub + 3*M_PI/2;
      float x = std::sin(theta) * std::cos(phi);
      float y = std::sin(theta) * std::sin(phi);
      float z = std::cos(thetaZ); // No pitch in camera ---->std::cos(theta);

      direction = octomap::point3d(x,y,z);

      for (int i = 0; i < numSteps; ++i) {
        octomap::point3d currentPoint; //= position + (i * voxelSize * direction);
        currentPoint.x() = currentPosition.x() + direction.x() * i * voxelSize;
        currentPoint.y() = currentPosition.y() + direction.y() * i * voxelSize;
        currentPoint.z() = currentPosition.z() + direction.z() * i * voxelSize;
        // receivedOctomap->coordToKeyChecked(currentPoint, key); // Find closest voxel to position
        // octomap::point3d closestVoxel = receivedOctomap->keyToCoord(key);
        // if (i==0 && j == 0){
        // ROS_INFO("Position:(%f,%f,%f); Direction:(%f,%f,%f); Phi: %f; Theta: %f; Orient: (%f,%f); Center: (%f,%f,%f)",p.position.x(),p.position.y(),p.position.z(),
        //           direction.x(),direction.y(),direction.z(),phi,theta,orient[0],orient[1],targetCenter[0],targetCenter[1],targetCenter[2]);
        // }
        // If position == target break
        int whichTarget;
        bool ifTarget = updateTrainData(currentPoint,target,gp.X_train_data,gp.Y_train_data,gp.num_data,whichTarget);
        if (ifTarget){
          //find distance between p.position and currentPoint
          break;
        }
        // std :: cout << "currentPoint: " << currentPoint << std::endl;
        octomap::OcTreeNode* node = receivedOctomap->search(currentPoint); // Retrieve the OcTreeNode for the current point
        // std::cout << "node: " << node << std::endl;
        if ((node && receivedOctomap->isNodeOccupied(node)) || i == numSteps-1) {
          // Voxel is occupied
          gp.X_train_data(gp.num_data,0) = currentPoint.x();
          gp.X_train_data(gp.num_data,1) = currentPoint.y();
          gp.X_train_data(gp.num_data,2) = currentPoint.z();
          gp.Y_train_data(gp.num_data) = 0;
          gp.num_data++;
          break;
        }
      }
    }
  }
  std::vector<double> params(2);
  params = PSO(10,10,gp);
  // params = {1,1};
// std::cout << "made it here: " << params[0] << " " << params[1] << std::endl;
// ROS_INFO("made it here: (%f)", gp_particle.Y_train.sum());
//   std::cout << "made it here: (" << gp_particle.Y_train.sum() << ")" << std::endl;
  if(gp.num_data>1 && gp.Y_train_data.sum()>1){
    clean_data(gp.X_train_data,gp.Y_train_data,gp.num_data);
    MatrixXd X_train_data = gp.X_train_data.block(0,0,gp.num_data,3);
    VectorXd y_train_data = gp.Y_train_data.head(gp.num_data);
    // std::cout << "num_data: " << gp.num_data << std::endl;
    predict(gp.X_train,X_train_data, y_train_data, gp.mu, params);  
  }
}
// Function to plot particles
visualization_msgs::MarkerArray createParticleMarkerArray(struct particle structureArray[], std::string world_frame_id)
{
    visualization_msgs::MarkerArray markerArray;
  
    for (size_t i = 0; i < numParticles+1; ++i)
    {
        // Create a marker for each structure
        visualization_msgs::Marker marker;
        marker.header.frame_id = world_frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "marker_array";
        marker.action = marker.DELETEALL;
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
      if (i == numParticles){
        marker.pose.position.x = globalBest.bestPosition.x();  // Assuming the x-coordinate is stored in the structure array
        marker.pose.position.y = globalBest.bestPosition.y();  // Assuming the y-coordinate is stored in the structure array
        marker.pose.position.z = globalBest.bestPosition.z();  // Assuming the z-coordinate is stored in the structure array
        // Set the color
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        // Set the scale
        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;
      
      } else{
        // Set the position
        marker.pose.position.x = structureArray[i].position.x();  // Assuming the x-coordinate is stored in the structure array
        marker.pose.position.y = structureArray[i].position.y();  // Assuming the y-coordinate is stored in the structure array
        marker.pose.position.z = structureArray[i].position.z();  // Assuming the z-coordinate is stored in the structure array
        // Set the color
        marker.color.r = 1.0;
        marker.color.g = 0.37;
        marker.color.b = 0.12;
        marker.color.a = 1.0;
        // Set the scale
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
      
      }
      
        // Set the orientation
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
      

        // Add the marker to the marker array
        markerArray.markers.push_back(marker);
    }
  
    return markerArray;
}
// -------------------------------------------------------------

// Function to plot target
visualization_msgs::MarkerArray createTargetMarkerArray(Array3D arr, std::string world_frame_id)
{
    visualization_msgs::MarkerArray markerArray;
    int rowCount = arr.size();
    for (int i = 0; i < rowCount; ++i)
    {
        // Create a marker for each target coord
        visualization_msgs::Marker marker;
        marker.header.frame_id = world_frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "marker_array";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
      
        marker.pose.position.x = arr[i][0];  // Assuming the x-coordinate is stored in the structure array
        marker.pose.position.y = arr[i][1];  // Assuming the y-coordinate is stored in the structure array
        marker.pose.position.z = arr[i][2];  // Assuming the z-coordinate is stored in the structure array
        // Set the color
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.5;
      
        // Set the orientation
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
      
        // Set the scale
        marker.scale.x = 0.1;
        marker.scale.y = targetRes;
        marker.scale.z = targetRes;
      
        // Add the marker to the marker array
        markerArray.markers.push_back(marker);
    }
  
    return markerArray;
}
// -------------------------------------------------------------

geometry_msgs::Pose createGoalMsg(){
  geometry_msgs::Pose goalMsg;
  tf::Quaternion quat;
  // Set best position
  goalMsg.position.x = globalBest.bestPosition.x();
  goalMsg.position.y = globalBest.bestPosition.y();
  goalMsg.position.z = globalBest.bestPosition.z();
  quat.setRPY(0,0,globalBest.bestYaw); // Set roll, pitch, yaw assuming static camera

  // Extract quaternions
  double qx = quat.x();
  double qy = quat.y();
  double qz = quat.z();
  double qw = quat.w();
  goalMsg.orientation.x = qx;
  goalMsg.orientation.y = qy;
  goalMsg.orientation.z = qz;
  goalMsg.orientation.w = qw;

  return goalMsg;

}
// -------------------------------------------------------------

// Function to publish target center
geometry_msgs::Pose publishTargetCenter(){
  geometry_msgs::Pose goalMsg;
  tf::Quaternion quat;
  // Set best position
  goalMsg.position.x = targetCenter[0];
  goalMsg.position.y = targetCenter[1];
  goalMsg.position.z = targetCenter[2];

  // Extract quaternions
  double qx = 0;
  double qy = 0;
  double qz = 0;
  double qw = 1;
  goalMsg.orientation.x = qx;
  goalMsg.orientation.y = qy;
  goalMsg.orientation.z = qz;
  goalMsg.orientation.w = qw;

  return goalMsg;

}
// -------------------------------------------------------------

// main function
int main(int argc, char** argv)
{
  // Initialize the ROS node
  ros::init(argc, argv, "meta_PSO");
  ros::NodeHandle nh("~");
  std::string octomap_default, pose_default, target_default, goal_center_default, world_frame_id;

  nh.param<std::string>("octomap_topic", octomap_default, "/octomap_binary");
  nh.param<std::string>("pose_topic", pose_default, "/firefly/ground_truth/pose");
  nh.param<std::string>("target_topic", target_default, "/best_particle_pose");
  nh.param<std::string>("goal_center_topic", goal_center_default, "/goal_center");
  nh.param<std::string>("world_frame_id", world_frame_id, "world");
  nh.param<double>("angle_offset", angle_offset, 0.0);
  nh.param<int>("target_width", ytargetSize, 1.0);
  nh.param<int>("target_height", ztargetSize, 1.0);
  nh.param<float>("target_start_in_y", leftY, 0.0);
  nh.param<float>("target_start_in_x", xTarget, 0.0);
  nh.param<float>("target_resolution", targetRes, 0.1);
  nh.param<double>("ray_length", rayLength, 10.0);
  nh.param<double>("field_of_view", fov, 1.0);

  // Subscribe to the octomap and pose topic
  ros::Subscriber octomap_sub = nh.subscribe<octomap_msgs::Octomap>(octomap_default, 10, octomapCallback);
  ros::Subscriber pose_sub = nh.subscribe<nav_msgs::Odometry>(pose_default, 1000, odomCallback);

  // Publish to marker array topic
  ros::Publisher markerArrayPub = nh.advertise<visualization_msgs::MarkerArray>("particle_swarm", 1);
  ros::Publisher targetArrayPub = nh.advertise<visualization_msgs::MarkerArray>("target_viz", 1);
  
  // Publish best particle
  ros::Publisher bestParticlePub = nh.advertise<geometry_msgs::Pose>(target_default,1, true);
  ros::Publisher goalCenterPub = nh.advertise<geometry_msgs::Pose>(goal_center_default,1);

  // Wait for the octomap to be received
  rows = ytargetSize*ztargetSize; cols = 3;
  increment = fov;
  phiIncrement = increment/(numRays);  // Angle increment in azimuthal direction
  thetaIncrement = increment/(numRays);  // Angle increment in polar direction
  makeTarget();
  calculateCenter();
 
  hyperplaneEquation(initX_train,A,B,C,D,centroid); 
  transformToNewHyperplane(initX_train,A,B,C,D,transformedPoints,
                      translationVector,rotationMatrix,centroid);
  while (ros::ok() && !receivedOctomap)
  {
    ros::spinOnce();
    ros::Duration(0.1).sleep();
  }
  // Calculate center of target and initialize particles
  initializeGP();
  initializeParticles();
  
// transformToNewHyperplane(gp.X_train,A,B,C,D,transformedPoints,
//                       translationVector,rotationMatrix,centroid);
  
 int iter = 1; double avg_time = 0;
//  Start while loop
  while (ros::ok()){
    // Initialize threads
    auto t1 = high_resolution_clock::now();
    

    std::vector<std::thread> threads;
    for (int i=0; i<numParticles; i++){
        threads.emplace_back(updateParticles,std::ref(globalBest),std::ref(particleArr[i]));
    }
    // std::cout << "Threads started" << std::endl;
    // Wait for all threads to finish
    for (std::thread& t : threads) {
        t.join();
    }
    // std::cout << "Threads done" << std::endl;
    // Update personal and global best
    mtx.lock();
    for (int i = 0; i < numParticles; ++i) {
        updateBest(particleArr[i]);
    }
    // ROS_INFO("BestScore: %f; Best Position: (%f,%f,%f)",globalBest.bestScore,globalBest.bestPosition.x(),globalBest.bestPosition.y(),globalBest.bestPosition.z());
    // std::cout << "Diff: " << abs(currentOrient.z()-wrapToPi(globalBest.bestYaw)) << " Orient: " << currentOrient.z() << " BestYaw: " << wrapToPi(globalBest.bestYaw) << std::endl;

    if ((currentPosition-globalBest.bestPosition).norm()<0.5 && abs(currentOrient.z()-wrapToPi(globalBest.bestYaw))<0.1){ 
      // ROS_INFO("Reached target!");
      // updateVehicleGP(gp);
      // std::cout << "gp_mu: " << gp.mu.sum() << " y_train: " << gp.Y_train.sum() << std::endl;
      // std::cout << "gp_mu_raw: " << gp.mu << std::endl;
      // updateTarget();
      // print all rows of target variable
      // for (int i = 0; i < target.size(); ++i) {
      //   std::cout << "new target" << std::endl;
      //   std::cout << target[i][0] << " " << target[i][1] << " " << target[i][2] << std::endl;
      // }
      // std::cout << "Target updated" << target << std::endl;
    }
    mtx.unlock();
        
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    avg_time += ms_double.count();
    // ROS_INFO("%f ms\n",avg_time/iter); //Time in milliseconds
    iter++;

    // Publish topics
    visualization_msgs::MarkerArray markerArray = createParticleMarkerArray(particleArr, world_frame_id);
    markerArrayPub.publish(markerArray);
    visualization_msgs::MarkerArray targetArray = createTargetMarkerArray(target, world_frame_id);
    targetArrayPub.publish(targetArray);
    geometry_msgs::Pose goalMsg = createGoalMsg();
    bestParticlePub.publish(goalMsg);
    geometry_msgs::Pose targetMsg = publishTargetCenter();
    goalCenterPub.publish(targetMsg);

    ros::spinOnce();
  }
  // Cleanup
  delete receivedOctomap;
  // delete tfListener;

  return 0;
}
