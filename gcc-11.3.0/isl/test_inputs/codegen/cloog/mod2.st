domain: "{ S1[i] : exists (e0 = floor((1 + i)/3): 3e0 <= i and 3e0 >= -1 + i and i >= 0 and i <= 3) }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i] -> [(i)] }]"
    options: "{ separate[i0] }"
