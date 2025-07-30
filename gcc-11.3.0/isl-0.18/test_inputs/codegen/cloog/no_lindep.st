domain: "[M, N] -> { S1[2 + N] }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S1[i0] -> [(1 + M)] }, { S1[i0] -> [(N)] }]"
    options: "[M, N] -> { separate[i0] }"
