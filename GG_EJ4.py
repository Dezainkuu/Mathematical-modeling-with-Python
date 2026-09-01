import numpy as np
import scipy.constants as constants

#Area of a triangle with input in cm and output in MKS

def triangle_area(base, height):
    area = (base * height) / 2
    return area * constants.centi**2  # Convert cm^2 to m^2

base = int(input("Enter the base of the triangle in cm: "))
height = int(input("Enter the height of the triangle in cm: "))

area = triangle_area(base, height)
print(f"The area of the triangle is {area} m^2")

#--------------------------------------------------------------------------------------------------------------------------------

#area of a triangle with three edges given in cm and output in MKS

def triangle_area2(edge1, edge2, edge3):
    s = (edge1 + edge2 + edge3) / 2
    area = (s * (s - edge1) * (s - edge2) * (s - edge3)) ** 0.5
    return area * constants.centi**2  # Convert cm^2 to m^2

edge1 = int(input("Enter the first edge of the triangle in cm: "))
edge2 = int(input("Enter the second edge of the triangle in cm: "))
edge3 = int(input("Enter the third edge of the triangle in cm: "))

area2 = triangle_area2(edge1, edge2, edge3)
print(f"The area of the triangle is {area2} m^2")

#--------------------------------------------------------------------------------------------------------------------------------

#area of a triangle with two edges and the angle between them given in cm and degrees, respectively, and output in MKS

def triangle_area3(edge1, edge2, angle):
    angle_rad = np.radians(angle)
    area = (edge1 * edge2 * np.sin(angle_rad)) / 2
    return area * constants.centi**2  # Convert cm^2 to m^2

edge1 = int(input("Enter the first edge of the triangle in cm: "))
edge2 = int(input("Enter the second edge of the triangle in cm: "))
angle = int(input("Enter the angle between the two edges in degrees: "))

area3 = triangle_area3(edge1, edge2, angle)
print(f"The area of the triangle is {area3} m^2")


#--------------------------------------------------------------------------------------------------------------------------------


