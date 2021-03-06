# shortest-path-finding-using-obstacle-avoidance
With an increase in manufacturing and automation industries, we see a big requirement of work being done without any manual labour. This has resulted in a hunt for smart robots and smart robotics. An Obstacle Avoidance Robot is one such intelligent robot, which can automatically sense and overcome obstacles on its path. A robot in general may come across various types of obstacles – broadly categorized into static and dynamic obstacles. The start and the destination points of the robot are predetermined – just as it is in most applications of robotics and automation. We will implement a machine learning algorithm known as Deep Reinforcement Learning. Through this learning algorithm we allow the robot to learn and dynamically avoid potentially harmful obstacles. Hence we intend to find the shortest and best path from the start to the destination while preventing the robot from stumbling onto various obstacles.
# Problem Definition
Obstacle avoidance is one of the most important aspects of mobile robotics. Without it, robot movement would be very restrictive and fragile. The need is to develop a system which allows us to sense objects to simultaneously avoid collisions at a very low cost and effective performance outcome.
There are, of course, several other interesting algorithms for obstacle avoidance. However, relatively few of them are suitable for real-time, embedded applications, novel kinematic algorithms for implementing planar hyper redundant manipulator obstacle avoidance, fuzzy logic solutions, virtual force field method, Spatial Nonlinear Model Predictive Control (NMPC) and many more.
Although various solutions already exist, these solutions are not devoid of problems of their own. Bug algorithm does not consider the actual kinematics of the robots, which is important to non-holonomic robots. It only considers the most recent sensor readings, and therefore sensor noise seriously affects the overall performance of the robot. This results in the robot to exhibit slow movement. The potential Field Algorithm performs poorly on narrow passages and is more difficult to make use of in real-time applications. The Vector Field Histogram (VFH) Algorithm involves a considerable computational load, which makes it difficult to implement on embedded systems. Some more problems are:
1. Trap situations due to local minima (cyclic behaviour)
2. No passage between closely spaced obstacles.
3. Oscillations in the presence of obstacles
4. Oscillations in narrow passages.

<p align="center">
  <img src="https://github.com/SuruchiParashar/shotest-path-finding-using-obstacle-avoidance/blob/master/ezgif.com-video-to-gif.gif" />
</p>
