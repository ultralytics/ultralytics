domain: "[m] -> { S2[coordT1, coordP1, 1 + coordT1, coordP1, 2 + coordT1 - coordP1, coordP1, 3 + coordT1 - coordP1, coordP1] : m >= 1 and coordT1 <= -3 + 2m and coordT1 >= 0 and coordP1 <= 1 + coordT1 and coordP1 <= m and coordP1 >= 3 - m + coordT1 and coordP1 >= 1; S4[coordT1, coordP1, 2 + coordT1, 1 + coordP1, 2 + coordT1 - coordP1, coordP1, 3 + coordT1 - coordP1, 1 + coordP1] : m >= 1 and coordT1 <= -4 + 2m and coordT1 >= 0 and coordP1 <= 1 + coordT1 and coordP1 <= -1 + m and coordP1 >= 3 - m + coordT1 and coordP1 >= 1; S6[coordT1, coordP1, 1 + coordT1, 1 + coordP1, 2 + coordT1 - coordP1, coordP1, 2 + coordT1 - coordP1, 1 + coordP1] : m >= 1 and coordT1 <= -3 + 2m and coordT1 >= 0 and coordP1 <= 1 + coordT1 and coordP1 <= -1 + m and coordP1 >= 2 - m + coordT1 and coordP1 >= 1; S1[coordT1, coordP1, 2 + coordT1 - coordP1, coordP1] : m >= 1 and coordP1 >= 2 - m + coordT1 and coordP1 <= 1 + coordT1 and coordP1 <= m and coordP1 >= 1; S8[coordT1, coordP1] : coordT1 <= -2 + 2m and coordT1 >= 0 and coordP1 <= 1 + coordT1 and coordP1 <= m and coordP1 >= 2 - m + coordT1 and coordP1 >= 1; S5[coordT1, coordP1, 1 + coordT1, coordP1, 2 + coordT1 - coordP1, coordP1, 3 + coordT1 - coordP1, coordP1] : m >= 1 and coordT1 <= -3 + 2m and coordT1 >= 0 and coordP1 <= 1 + coordT1 and coordP1 <= m and coordP1 >= 3 - m + coordT1 and coordP1 >= 1; S7[coordT1, coordP1, 2 + coordT1, 1 + coordP1, 2 + coordT1 - coordP1, coordP1, 3 + coordT1 - coordP1, 1 + coordP1] : m >= 1 and coordT1 <= -4 + 2m and coordT1 >= 0 and coordP1 <= 1 + coordT1 and coordP1 <= -1 + m and coordP1 >= 3 - m + coordT1 and coordP1 >= 1; S3[coordT1, coordP1, 1 + coordT1, 1 + coordP1, 2 + coordT1 - coordP1, coordP1, 2 + coordT1 - coordP1, 1 + coordP1] : m >= 1 and coordT1 <= -3 + 2m and coordT1 >= 0 and coordP1 <= 1 + coordT1 and coordP1 <= -1 + m and coordP1 >= 2 - m + coordT1 and coordP1 >= 1 }"
child:
  context: "[m] -> { [] : m >= 0 }"
  child:
    schedule: "[m] -> [{ S7[i0, i1, i2, i3, i4, i5, i6, i7] -> [(1 + i0)]; S4[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i0)]; S8[i0, i1] -> [(i0)]; S3[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i0)]; S1[i0, i1, i2, i3] -> [(i0)]; S2[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i0)]; S5[i0, i1, i2, i3, i4, i5, i6, i7] -> [(1 + i0)]; S6[i0, i1, i2, i3, i4, i5, i6, i7] -> [(1 + i0)] }]"
    options: "[m] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[m] -> { S2[i0, i1, i2, i3, i4, i5, i6, i7]; S6[i0, i1, i2, i3, i4, i5, i6, i7]; S4[i0, i1, i2, i3, i4, i5, i6, i7]; S1[i0, i1, i2, i3]; S7[i0, i1, i2, i3, i4, i5, i6, i7]; S5[i0, i1, i2, i3, i4, i5, i6, i7]; S3[i0, i1, i2, i3, i4, i5, i6, i7] }"
        child:
          schedule: "[m] -> [{ S7[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i3)]; S4[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i1)]; S3[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i1)]; S1[i0, i1, i2, i3] -> [(i1)]; S2[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i1)]; S5[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i3)]; S6[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i3)] }]"
          options: "[m] -> { separate[i0] }"
          child:
            sequence:
            - filter: "[m] -> { S6[i0, i1, i2, i3, i4, i5, i6, i7]; S5[i0, i1, i2, i3, i4, i5, i6, i7]; S7[i0, i1, i2, i3, i4, i5, i6, i7] }"
              child:
                schedule: "[m] -> [{ S7[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)]; S5[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)]; S6[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)] }, { S7[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)]; S5[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)]; S6[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)] }]"
                options: "[m] -> { separate[i0] }"
            - filter: "[m] -> { S1[i0, i1, i2, i3] }"
            - filter: "[m] -> { S2[i0, i1, i2, i3, i4, i5, i6, i7]; S4[i0, i1, i2, i3, i4, i5, i6, i7]; S3[i0, i1, i2, i3, i4, i5, i6, i7] }"
              child:
                schedule: "[m] -> [{ S4[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)]; S3[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)]; S2[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)] }, { S4[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)]; S3[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)]; S2[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)] }]"
                options: "[m] -> { separate[i0] }"
      - filter: "[m] -> { S8[i0, i1] }"
