domain: "[t] -> { S1[i0, t] : exists (e0 = floor((t - i0)/16): 16e0 = t - i0 and i0 >= 0 and i0 <= 99 and t >= 0 and t <= 15) }"
child:
  context: "[t] -> { [] }"
  child:
    schedule: "[t] -> [{ S1[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)] }]"
    options: "[t] -> { separate[i0] }"
