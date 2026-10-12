import numpy as np

# ACC Values
desired_distance = 40
actual_distance = 30
our_speed = 80
lead_speed = 60

# PID gains
Kp = 0.5
Ki = 0.1
Kd = 0.2

# PID assumptions
accumulated_errors = 5
rate_of_error = 2

# Distance error
error = desired_distance - actual_distance

# PID Calculation
P = Kp * error
I = Ki * accumulated_errors
D = Kd * rate_of_error

# Total PID Output
PID_output = P + I + D

# Adjust Vehicle Speed
our_speed = our_speed - PID_output

print("Distance Error =", error, "m")
print("P =", P)
print("I =", I)
print("D =", D)
print("PID Output =", PID_output)
print("Lead Vehicle Speed =", lead_speed, "km/h")
print("Our New Speed =", our_speed, "km/h")