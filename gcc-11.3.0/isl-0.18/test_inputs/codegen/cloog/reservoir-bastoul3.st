domain: "{ S1[i0, i1, i2] : 2i2 = i0 - i1 and i1 >= 1 and i1 <= 3 and i1 <= -2 + i0 and i1 >= -6 + i0 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i0, i1, i2] -> [(i0)] }, { S1[i0, i1, i2] -> [(i1)] }, { S1[i0, i1, i2] -> [(i2)] }]"
    options: "{ separate[i0] }"
