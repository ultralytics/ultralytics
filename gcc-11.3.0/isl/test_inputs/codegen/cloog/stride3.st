domain: "[m, n] -> { S1[i] : i >= 1 and i <= n and i >= m }"
child:
  context: "[m, n] -> { [] }"
  child:
    schedule: "[m, n] -> [{ S1[i0] -> [(50i0)] }]"
    options: "[m, n] -> { separate[i0] }"
