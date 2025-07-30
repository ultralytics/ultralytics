domain: "{ S1[i0] : exists (e0 = floor((1 + i0)/3): 3e0 <= i0 and 3e0 >= -1 + i0 and i0 >= 0 and i0 <= 3) }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i0] -> [(i0)] }]"
    options: "{ separate[i0] }"
