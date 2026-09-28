import numpy as np

def test_function():
    # Example test function that uses numpy
    arr = np.array([1, 2, 3])
    assert np.sum(arr) == 6, "Sum of array should be 6"
    print("Test passed!")

def adjust_yaw(self, yaw: float) -> float:
    """Adjust yaw angle to account for full rotations and return the adjusted yaw.

    This function keeps track of the number of full rotations both clockwise and counterclockwise, and adjusts the yaw angle accordingly so that it reflects the absolute angle in radians. It ensures that the yaw angle is not wrapped around to the range of -pi to pi, but instead accumulates the full rotations.
    This is particularly useful for applications where the absolute orientation of the vehicle is important, such as in control algorithms or navigation systems.
    The function also initializes the first yaw value and keeps track of the previous yaw value to determine if a full rotation has occurred.

    Args:
        yaw (float): The yaw angle in radians from the motion capture system after being converted from quaternion to euler angles.

    Returns:
        psi (float): The adjusted yaw angle in radians, accounting for full rotations.
    """        
    mocap_psi = yaw

    if not self.mocap_initialized:
        self.mocap_initialized = True
        self.prev_mocap_psi = mocap_psi
        self.unwrapped_psi = mocap_psi
        return mocap_psi

    # MoCap angles are from -pi to pi, whereas the angle state variable should be an absolute angle (i.e. no modulus wrt 2*pi).
    # Accumulate the WRAPPED increment between consecutive samples (like numpy.unwrap). For normal motion this equals the
    # old "count crossings of +-0.9*pi" rule, but it stays correct when a sample jumps across the wrap by less than that
    # band (e.g. +170 deg -> -100 deg during an upset), which the old rule turned into a spurious ~2*pi error.
    delta = (mocap_psi - self.prev_mocap_psi + np.pi) % (2 * np.pi) - np.pi
    self.unwrapped_psi += delta
    self.prev_mocap_psi = mocap_psi
    self.full_rotations = int(round((self.unwrapped_psi - mocap_psi) / (2 * np.pi))) # kept for compatibility

    return self.unwrapped_psi



if __name__ == "__main__":
    test_function()