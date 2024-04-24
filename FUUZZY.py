def triangular_membership(x, a, b, c):
  if x <= a or x >= c:
      return 0
  elif a < x <= b:
      return (x - a) / (b - a)
  elif b < x < c:
      return (c - x) / (c - b)

def fuzzy_rule(rule, temp, pressure):
    degreeOfMemberShip = []
    # Fuzzy logic rule for temperature
    f1 = rule[0]
    if f1 == "temp_low":
        temp_low = triangular_membership(temp, 10,10,25)
        degreeOfMemberShip.append(temp_low)
    elif f1 == "temp_bl_avg":
        temp_bl_avg = triangular_membership(temp, 15, 30, 45)
        degreeOfMemberShip.append(temp_bl_avg)
    elif f1 == "temp_avg":
        temp_avg = triangular_membership(temp, 40, 50, 60)
        degreeOfMemberShip.append(temp_avg)
    elif f1 == "temp_ab_avg":
        temp_ab_avg = triangular_membership(temp, 55, 70, 85)
        degreeOfMemberShip.append(temp_ab_avg)
    elif f1 == "temp_high":
        temp_high = triangular_membership(temp, 75, 90, 105)
        degreeOfMemberShip.append(temp_high)

    # Fuzzy logic rule for pressure
    f2 = rule[1]
    if f2 == "pressure_low":
        pressure_low = triangular_membership(pressure, 0.25,1,1.75)
        degreeOfMemberShip.append(pressure_low)
    elif f2 == "pressure_bl_avg":
        pressure_bl_avg = triangular_membership(pressure, 1.25,2,2.75)
        degreeOfMemberShip.append(pressure_bl_avg)
    elif f2 == "pressure_avg":
        pressure_avg = triangular_membership(pressure, 2,3,4)
        degreeOfMemberShip.append(pressure_avg)
    elif f2 == "pressure_ab_avg":
        pressure_ab_avg = triangular_membership(pressure, 3.25,4,4.75)
        degreeOfMemberShip.append(pressure_ab_avg)
    elif f2 == "pressure_high":
        pressure_high = triangular_membership(pressure, 4.25,5,5.75)
        degreeOfMemberShip.append(pressure_high)
    print(f"Rule 1 : {degreeOfMemberShip}")
    return min(degreeOfMemberShip)

def area(z):
  area = z["c"] - z["l"]
  return area

def consequent(c,z1,z2):
  cons = None
  volumes = []
  htpw_low = {"l": 0.25, "c": 1, "r":1.75}
  htpw_bl_avg = {"l": 1.25, "c": 2, "r":2.75}
  htpw_avg = {"l": 2, "c": 3, "r":4}
  htpw_ab_avg = {"l": 3.25, "c": 4, "r":4.75}
  htpw_high = {"l": 4.25, "c": 5, "r":5.75}

  valve_low = {"l": 0.25, "c": 1, "r":1.75}
  valve_med_low = {"l": 1.25, "c": 2, "r":2.75}
  valve_med = {"l": 2, "c": 3, "r":4}
  valve_med_high = {"l": 3.25, "c": 4, "r":4.75}
  valve_high = {"l": 4.25, "c": 5, "r":5.75}


  for i in range(len(c)):
    if c[i] == "htpw_low":
      v = area(htpw_low)
      volumes.append(v)
    elif c[i] == "htpw_bl_avg":
      v = area(htpw_bl_avg)
      volumes.append(v)
    elif c[i] == "htpw_avg":
      v = area(htpw_avg)
      volumes.append(v)
    elif c[i] == "htpw_ab_avg":
      v = area(htpw_ab_avg)
      volumes.append(v)
    elif c[i] == "htpw_high":
      v = area(htpw_high)
      volumes.append(v)
    elif c[i] == "valve_low":
        v = area(valve_low)
        volumes.append(v)
    elif c[i] == "valve_med_low":
        v = area(valve_med_low)
        volumes.append(v)
    elif c[i] == "valve_med":
        v = area(valve_med)
        volumes.append(v)
    elif c[i] == "valve_med_high":
        v = area(valve_med_high)
        volumes.append(v)
    elif c[i] == "valve_high":
        v = area(valve_high)
        volumes.append(v)
  if c[0].startswith("htpw"):
    print(f"v1: {volumes[0]} , v2: {volumes[1]}")
    cons = ((z1 * volumes[0] * htpw_ab_avg["c"] + z2 * volumes[1] * htpw_high["c"]) / (z1 * volumes[0] + z2 * volumes[1]))
  elif c[0].startswith("valve"):
    print(f"v1: {volumes[0]} , v2: {volumes[1]}")
    cons = ((z1 * volumes[0] * valve_med_low["c"] + z2 * volumes[1] * valve_low["c"]) / (z1 * volumes[0] + z2 * volumes[1]))
  else:
    cons = 0
  return cons

temperature = float(input("Enter Temperature: "))
pressure = float(input("Enter Pressure: "))
anticident_1 = ["temp_bl_avg", "pressure_bl_avg"]
anticident_2 = ["temp_low", "pressure_low"]
z1 = fuzzy_rule(anticident_1, temperature, pressure)
z2 = fuzzy_rule(anticident_2, temperature, pressure)
print(f"Fuzzyfication \n\nZ1: {z1} & Z2: {z2}\n")
c1 = ["htpw_ab_avg", "htpw_high"]
crisp_output_for_HeatingPower = consequent(c1,z1,z2)
print(f"Crips Output for Heating Power: {crisp_output_for_HeatingPower}")
c2 = ["valve_med_low", "valve_low"]
crisp_output_for_ValveOpen = consequent(c2,z1,z2)
print(f"Crips Output for Valve Opening: {crisp_output_for_ValveOpen}")
