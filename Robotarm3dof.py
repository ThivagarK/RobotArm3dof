import RPi.GPIO as GPIO
from time import sleep
import numpy as np
from matplotlib import pyplot as plt
import os

#GPIO.setmode(GPIO.BOARD)

#set pin 3 for PWM
#GPIO.setup(3, GPIO.OUT)
#pwm1=GPIO.PWM(3, 50)
#pwm1.start(0)

#set pin 5 for PWM
#GPIO.setup(5, GPIO.OUT)
#pwm2=GPIO.PWM(5, 50)
#pwm2.start(0)

#set pin 12 for PWM
#GPIO.setup(12, GPIO.OUT)
#pwm3=GPIO.PWM(12, 50)
#pwm3.start(0)

speed = 1

class Servomotor:
    def __init__(self, name, pin):
        self.name = name
        self.pin = pin

# This function returns the angle between a and b 
def angle_from_dot_product(a, b):
    ax, ay, az = a
    bx, by, bz = b

    a_mag = np.sqrt(np.power(ax, 2) + np.power(ay, 2) + np.power(az, 2))
    b_mag = np.sqrt(np.power(bx, 2) + np.power(by, 2) + np.power(bz, 2))

    theta = np.arccos((1 / (a_mag * b_mag)) * (ax * bx + ay * by + az * bz))

    return theta

# This function converts theta,d,a, and alpha into dh parameters
def dh(theta, d, a, alpha):
    A11 = np.cos(theta)
    A12 = -np.cos(alpha) * np.sin(theta)
    A13 = np.sin(alpha) * np.sin(theta)
    A14 = a * np.cos(theta)

    A21 = np.sin(theta)
    A22 = np.cos(alpha) * np.cos(theta)
    A23 = -np.sin(alpha) * np.sin(theta)
    A24 = a * np.sin(theta)

    A31 = 0
    A32 = np.sin(alpha)
    A33 = np.cos(alpha)
    A34 = d

    A41 = 0
    A42 = 0
    A43 = 0
    A44 = 1

    A = np.array([[A11, A12, A13, A14],  
                  [A21, A22, A23, A24],  
                  [A31, A32, A33, A34], 
                  [A41, A42, A43, A44]])

    return A

# Projection step used in FABRIK
def project_along_vector(x1, y1, z1, x2, y2, z2, L):
    vx = x2 - x1
    vy = y2 - y1
    vz = z2 - z1
    v = np.sqrt(np.power(vx, 2) + np.power(vy, 2) + np.power(vz, 2))

    ux = vx / v
    uy = vy / v
    uz = vz / v

    px = x1 + L * ux
    py = y1 + L * uy
    pz = z1 + L * uz

    return np.array([px, py, pz])

def fabrik(l1, l2, l3, x_prev, y_prev, z_prev, x_command, y_command, z_command, tol_limit, max_iterations):
    # Initializing the base rotation
    q1_prev = np.arctan2(y_prev[3], x_prev[3])
    q1 = np.arctan2(y_command, x_command)
    base_rotation = q1 - q1_prev

    R_z = np.array([[np.cos(base_rotation), -np.sin(base_rotation), 0.0],
                    [np.sin(base_rotation), np.cos(base_rotation), 0.0],
                    [0.0, 0.0, 1.0]])

    p3 = np.dot(R_z, [x_prev[2], y_prev[2], z_prev[2]])
    p2 = np.dot(R_z, [x_prev[1], y_prev[1], z_prev[1]])
    p1 = np.dot(R_z, [x_prev[0], y_prev[0], z_prev[0]])

    p3x, p3y, p3z = p3[0], p3[1], p3[2]
    p2x, p2y, p2z = p2[0], p2[1], p2[2]
    p2x_o, p2y_o, p2z_o = p2[0], p2[1], p2[2]
    p1x, p1y, p1z = p1[0], p1[1], p1[2]

    tol = tol_limit + 1
    iterations = 0

    if np.sqrt(np.power(x_command, 2) + np.power(y_command, 2) + np.power(z_command, 2)) > (l1 + l2 + l3):
        # Clear the command terminal
        os.system('cls' if os.name == 'nt' else 'clear')
        print('Desired point is out of reach. Please enter the positions again.')
        main()
    else:
        while tol > tol_limit and iterations < max_iterations:

            # Backward pass
            p4x, p4y, p4z = x_command, y_command, z_command
            p3x, p3y, p3z = project_along_vector(p4x, p4y, p4z, p3x, p3y, p3z, l3)
            p2x, p2y, p2z = project_along_vector(p3x, p3y, p3z, p2x, p2y, p2z, l2)

            # Forward pass
            p3x, p3y, p3z = project_along_vector(p2x_o, p2y_o, p2z_o, p3x, p3y, p3z, l2)
            p4x, p4y, p4z = project_along_vector(p3x, p3y, p3z, p4x, p4y, p4z, l3)

            tolx = p4x - x_command
            toly = p4y - y_command
            tolz = p4z - z_command

            tol = np.sqrt(np.power(tolx, 2) + np.power(toly, 2) + np.power(tolz, 2))
            p_joints = np.array([[p1x, p2x, p3x, p4x],
                                 [p1y, p2y, p3y, p4y],
                                 [p1z, p2z, p3z, p4z]])

            end_point = np.array([p4x, p4y, p4z])

            v21 = np.array([p3x - p2x, p3y - p2y, p3z - p2z])
            v32 = np.array([p4x - p3x, p4y - p3y, p4z - p3z])

            q2 = np.arctan2((p3z - p2z), np.sqrt(np.power(p3x - p2x, 2) + np.power(p3y - p2y, 2)))
            q3 = -1 * angle_from_dot_product(v21, v32)

            q_joints = np.array([q1, q2, q3])
            iterations += 1

    print("Number of iterations: " + str(iterations))
    return q_joints, tol, end_point, p_joints

# Forward kinematics to compute joint positions based on angles
def forward_kinematics(l1, l2, l3, l4, q1, q2, q3):
    T10 = dh(0, l1, 0, 0)  # Base rotation relative to global frame
    T21 = dh(q1, l2, 0, np.pi / 2)
    T32 = dh(q2, 0, l3, 0)
    T43 = dh(q3, 0, l4, np.pi / 2)

    T20 = np.dot(T10, T21)
    T30 = np.dot(T20, T32)
    T40 = np.dot(T30, T43)

    p_joints = [T10[0:3, 3], T20[0:3, 3], T30[0:3, 3], T40[0:3, 3]]
    
    p_joints = np.array(p_joints).T
    return p_joints

# Conversion from degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi / 180)

def rad_to_deg(angle):
    return angle * (180 / np.pi)

# Robot arm parameters function with joint angles in degrees
def Robotarm_parameter():
    angle = [90, 40, -90]
    length = [4, 2, 8, 8]
    return angle, length

def command_position():
    x= int(input('Enter your servo1 position: '))
    y= int(input('Enter your servo2 position: '))
    z= int(input('Enter your servo3 position: '))
    position = [x,y,z]
    return position

#def Set_Angle(angle):
    duty= []
    for x in angle:
        value = x / 18 + 2
        duty.append(value)
    return duty

#def SetAngle(pin, angle, previous_angle):
    i = 0
    angle = int(angle)
    previous_angle = int(previous_angle)
    if previous_angle < angle:
        for i in range(previous_angle, angle+5, 5):
            duty = i / 18 + 2
            if pin == 3:
                #pwm1.ChangeDutyCycle(duty)
                print(duty)
                print("Current {} motor position: {}".format(servo1.name,i))
            elif pin == 5:
                #pwm2.ChangeDutyCycle(duty)
                print(duty)
                print("Current {} motor position: {}".format(servo2.name,i))
            elif pin == 12:
                #pwm3.ChangeDutyCycle(duty)
                print(duty)
                print("Current {} motor position: {}".format(servo3.name,i))
            sleep(1)

    elif previous_angle > angle:
        for i in range(previous_angle,angle-5,-5):
            duty = i / 18 + 2
            if pin == 3:
                #pwm1.ChangeDutyCycle(duty)
                print(duty)
                print("Current {} motor position: {}".format(servo1.name,i))
            elif pin == 5:
                #pwm2.ChangeDutyCycle(duty)
                print(duty)
                print("Current {} motor position: {}".format(servo2.name,i))
            elif pin == 12:
                #pwm3.ChangeDutyCycle(duty)
                print(duty)
                print("Current {} motor position: {}".format(servo3.name,i))
            sleep(1)
    else:
        i = angle
        print ("Motor position remains!!!")
    return i

#def SetAngle(pin, angle):
    duty = angle / 18 + 2
    if pin == 3:
        #pwm1.ChangeDutyCycle(duty)
        print(duty)
        print("Current {} motor position: {}".format(servo1.name,angle))
    elif pin == 5:
        #pwm2.ChangeDutyCycle(duty)
        print(duty)
        print("Current {} motor position: {}".format(servo2.name,angle))
    elif pin == 12:
        #pwm3.ChangeDutyCycle(duty)
        print(duty)
        print("Current {} motor position: {}".format(servo3.name,angle))
    sleep(1)
    
#def move_robot(angle):
    angle_degree = rad_to_deg(angle)
    angle_degree[0] = round(angle_degree[0],2) + 90
    angle_degree[1] = round(angle_degree[1],2) + 18.66
    angle_degree[2] = round(angle_degree[2],2) + 257.74
    if angle_degree[0] <= 180 and angle_degree[0]>=0:
        if angle_degree[1] >= 0 and angle_degree[1] <= 40:
            if angle_degree[2] >= 175:
                SetAngle(servo2.pin, angle_degree[1])
                SetAngle(servo3.pin, angle_degree[2])
                SetAngle(servo1.pin, angle_degree[0])
            elif angle_degree[2] >= 135 and angle_degree[2] < 175:
                SetAngle(servo3.pin, 175)
                SetAngle(servo2.pin, angle_degree[1])
                SetAngle(servo1.pin, angle_degree[0])
        elif angle_degree[1] > 40 and angle_degree[1] <=90:
            if angle_degree[2] <= 175 and angle_degree[2] >= 80:
                SetAngle(servo1.pin, angle_degree[0])
                SetAngle(servo2.pin, angle_degree[1])
                SetAngle(servo3.pin, angle_degree[2])
        else:
            SetAngle(servo1.pin, 90)
            SetAngle(servo2.pin, 0)
            SetAngle(servo3.pin, 135)
            print("Beyond limit")
            #main()
    return angle_degree

def SetAngle(angle, previous_angle):
    #servomotor pins
    servo_pins=[3, 5, 12]
    servo_name=["Base", "Arm", "Hand"]

    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)

    # Set the GPIO pins for PWM
    pwm_channels = []
    for pin in servo_pins:
        GPIO.setup(pin, GPIO.OUT)
        pwm_channels.append(GPIO.PWM(pin, 50))  # 50Hz for servos

    # Start PWM on all servos
    for pwm in pwm_channels:
        pwm.start(0)  # Start PWM with 0 duty cycle (inactive)

    # Run the loop until all values match
    while previous_angle[0] != angle[0] or previous_angle[1] != angle[1] or previous_angle[2] != angle[2]:
        # Adjust Base servo
        if previous_angle[0] < angle[0]:
            previous_angle[0] += speed
        elif previous_angle[0] > angle[0]:
            previous_angle[0] -= speed

        # Adjust Arm servo
        if previous_angle[1] < angle[1]:
            previous_angle[1] += speed
        elif previous_angle[1] > angle[1]:
            previous_angle[1] -= speed

        # Adjust Hand servo
        if previous_angle[2] < angle[2]:
            previous_angle[2] += speed
        elif previous_angle[2] > angle[2]:
            previous_angle[2] -= speed
        
        # Update PWM duty cycle for each servo based on the angle
        for i, pwm in enumerate(pwm_channels):
            duty = previous_angle[i] / 18 + 2  # Calculate duty cycle (for 180° servos typically)
            pwm.ChangeDutyCycle(duty)  # Change the PWM duty cycle for the servo
            print(f"{servo_name[i]} servomotor Duty cycle: {duty}% for angle: {previous_angle[i]}°")

        # Optional: Add a small delay to make the servo movement visible
        sleep(0.1)  # Sleep for 100ms to slow down the servo movement

    # Stop PWM after reaching the target angle
    #for pwm in pwm_channels:
        #pwm.stop()  # Stop PWM on the pin

    # Cleanup GPIO setup
    #GPIO.cleanup()

    return previous_angle


def move_robot(angle,previous_angle):
    angle_degree = rad_to_deg(angle)
    angle_degree[0] = round(angle_degree[0])
    angle_degree[1] = round(angle_degree[1] + 18.66)
    angle_degree[2] = round(angle_degree[2] + 257.74)
    print("Angle of servomotor to move {}".format(angle_degree))
    if angle_degree[0] <= 180 and angle_degree[0]>=0:
        if angle_degree[1] >= 0 and angle_degree[1] <= 40:
            if angle[2] >= 175:
                previous_angle = SetAngle([angle_degree[0],angle_degree[1],175], previous_angle)
                #current_angle2 = SetAngle(servo2.pin, angle_degree[1], previous_angle[1])
                #current_angle1 = SetAngle(servo1.pin, angle_degree[0], previous_angle[0])
                #current_angle = SetAngle(servo1.pin, angle_degree, previous_angle)
            elif angle_degree[2] >= 135 and angle_degree[2] < 175:
                previous_angle = SetAngle(angle_degree, previous_angle)
                #current_angle3 = SetAngle(servo3.pin, angle_degree[2] , previous_angle[2])
                #current_angle2 = SetAngle(servo2.pin, angle_degree[1], previous_angle[1])
                #current_angle1 = SetAngle(servo1.pin, angle_degree[0], previous_angle[0])
        elif angle_degree[1] > 40 and angle_degree[1] <=90:
            if angle_degree[2] <= 175 and angle_degree[2] >= 80:
                previous_angle = SetAngle(angle_degree, previous_angle)
                #current_angle1 = SetAngle(servo1.pin, angle_degree[0], previous_angle[0])
                #current_angle2 = SetAngle(servo2.pin, angle_degree[1], previous_angle[1])
                #current_angle3 = SetAngle(servo3.pin, angle_degree[2], previous_angle[2])
            else:
                current_angle3 = angle_degree[2]
                current_angle2 = angle_degree[1]
                current_angle1 = angle_degree[0]
                #SetAngle(servo1.pin, 90, previous_angle[0])
                #SetAngle(servo2.pin, 0, previous_angle[1])
                #SetAngle(servo3.pin, 135, previous_angle[2])
                SetAngle([90, 0, 135], previous_angle)
                print("Beyond limit")
                #main()
                previous_angle = [current_angle1,current_angle2,current_angle3]

    return previous_angle

def plot_robot_arm(x_joints_o, y_joints_o, z_joints_o, x_IK, y_IK, z_IK, p1_new, p2_new, p3_new, x_command, y_command, z_command):
    plt.style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.plot(x_joints_o, y_joints_o, z_joints_o, color='blue')
    ax1.scatter(x_joints_o, y_joints_o, z_joints_o, label='Start', color='blue')

    ax1.plot(x_IK, y_IK, z_IK, color='red')
    ax1.scatter(x_IK, y_IK, z_IK, color='red', label='IK Position')

    ax1.scatter(x_command, y_command, z_command, color='green', label='Commanded Position')

    # plot fabrik position
    ax1.plot(p1_new,p2_new,p3_new,color='yellow')
    ax1.scatter(p1_new,p2_new,p3_new,label='Fabrik',color='yellow')

    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")

    plt.legend()
    plt.show()

def dataset(length, angle, p_joints, tol_limit, tolerance, end_point, new_point, command_point):
    new_joint_points = forward_kinematics(length[0], length[1], length[2], length[3], angle[0], angle[1], angle[2])

    print("The actual tolerance is " + str(tol_limit) + " and the error of the Fabrik equation is " + str(tolerance))
    print("Fabrik equation result: " + str(end_point))
    print("Forward kinematics result: " + str(new_joint_points[:, 3]))
    print(rad_to_deg(angle))
    plot_robot_arm(p_joints[0], p_joints[1], p_joints[2], new_joint_points[0], new_joint_points[1], new_joint_points[2], new_point[0], new_point[1], new_point[2], command_point[0], command_point[1], command_point[2])

def main():

    joint, length = Robotarm_parameter()
    q1 = deg_to_rad(joint[0])  # Base
    q2 = deg_to_rad(joint[1])  # First link relative to base
    q3 = deg_to_rad(joint[2])  # Second link relative to first link

    l1, l2, l3, l4 = length[0], length[1], length[2] , length[3]

    tol_limit = l2 / 100  # tolerance limit
    max_iterations = 100  # max iterations allowed

    p_joints = forward_kinematics(length[0], length[1], length[2] , length[3], q1, q2, q3)

    x_joints_o, y_joints_o, z_joints_o = p_joints[0], p_joints[1], p_joints[2]

    # Initialisation.
    #initial_angle = [90,0,135]
    #previous_angle = move_robot(np.array([q1,q2,q3]))

    # Initialisation.
    initial_angle = np.array([q1,q2,q3])
    previous_angle = move_robot(initial_angle,previous_angle = [45,45,175])
    print("Initialisation complete: {}".format(previous_angle))

    #q1_new, q2_new, q3_new = new_q
    #p1_new, p2_new, p3_new = new_p

    # Function to move the robot.
    # new_angle = move_robot(new_q)

    try:
        while True:
            position = command_position()
            angle, tolerance, end_point, new_p = fabrik(l1, l3, l4, x_joints_o, y_joints_o, z_joints_o, position[0], position[1], position[2], tol_limit, max_iterations)
            previous_angle = move_robot(angle,previous_angle)
            print("Movement complete: {}".format(previous_angle))
            dataset(length, angle, p_joints, tol_limit, tolerance, end_point, new_p, position)
#            previous_angle = move_robot(np.array([angle[0],deg_to_rad(45-18.66),deg_to_rad(135-257.74)]),previous_angle)
#            print("Homing position: {}".format(previous_angle))
    
    except KeyboardInterrupt:
        print("Program Stopped")
        #pwm1.stop()
        #pwm2.stop()
        #pwm3.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    
    #Assign pin out
    servo1 = Servomotor("Base", 3)
    servo2 = Servomotor("Arm", 5)
    servo3 = Servomotor("Hand", 12)

    main()
