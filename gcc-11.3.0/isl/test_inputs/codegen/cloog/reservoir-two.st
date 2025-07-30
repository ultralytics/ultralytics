domain: "{ S1[i0, i1, i2] : 2i1 = 3 - i0 and 2i2 = 9 + i0 and i0 >= 0 and i0 <= 1 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i0, i1, i2] -> [(i0)] }, { S1[i0, i1, i2] -> [(i1)] }, { S1[i0, i1, i2] -> [(i2)] }]"
    options: "{ separate[i0] }"
