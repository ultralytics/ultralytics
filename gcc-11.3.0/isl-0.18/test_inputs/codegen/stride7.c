for (int c0 = 2; c0 <= 200; c0 += 64) {
  for (int c2 = c0 - 1; c2 <= 120; c2 += 1)
    s2(c0, c2);
  for (int c2 = 122; c2 <= c0 + 62; c2 += 1)
    s4(c0, c2);
}
