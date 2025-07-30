domain: "{ S1[i0, i1, i2, i3] : i3 <= 4 - 2i0 - 2i1 and i3 >= i2 and i2 <= 9 - 2i0 and i2 >= 0 and i2 >= 1 - 2i0 and i3 <= 1 + i2 and i2 <= 1 and i3 >= 1 - 2i0 - 2i1 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i0, i1, i2, i3] -> [(2i0)] }]"
    options: "{ separate[i0] }"
    child:
      schedule: "[{ S1[i0, i1, i2, i3] -> [(2i0 + 2i1)] }]"
      options: "{ separate[i0] }"
      child:
        schedule: "[{ S1[i0, i1, i2, i3] -> [(2i0 + i2)] }]"
        options: "{ separate[i0] }"
        child:
          schedule: "[{ S1[i0, i1, i2, i3] -> [(2i0 + 2i1 + i3)] }]"
          options: "{ separate[i0] }"
