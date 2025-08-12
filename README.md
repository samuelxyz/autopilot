# autopilot

This started as a sandbox project where I wanted to just play around with control systems and sensors. I am now using it as an exercise to learn and implement Kalman filtering for a spacecraft navigation system.

![](assets/EKF_Test_Gyro_StarTracker.png)

This graph shows an early test using an Extended Kalman Filter collecting data from (1) a continuously running gyroscope and (2) a star tracker that produces readings once per second. (I don't think real star trackers can easily take readings while the spacecraft has significant angular velocity but it's for the sake of the filter testing here)