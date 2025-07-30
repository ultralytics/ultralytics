domain: "[n] -> { S1[ii, i] : i >= 0 and i <= n and i <= 9 + 10ii and i >= 10ii }"
child:
  context: "[n] -> { [] : n >= 0 }"
  child:
    schedule: "[n] -> [{ S1[ii, i] -> [(ii)] }, { S1[ii, i] -> [(i)] }]"
    options: "[n] -> { separate[i0] }"
