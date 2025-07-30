domain: "{ S1[i0, i1, i2] : 2i1 = -1 + i0 and 3i2 = -2 + i0 and i0 >= 0 and i0 <= 100 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i0, i1, i2] -> [(i0)] }, { S1[i0, i1, i2] -> [(i1)] }, { S1[i0, i1, i2] -> [(i2)] }]"
    options: "{ separate[i0] }"
