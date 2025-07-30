for (int c0 = 1; c0 <= 11; c0 += 1) {
  for (int c1 = max(1, c0 - 3); c1 <= min(c0, -c0 + 8); c1 += 1)
    s1(c1, c0 - c1 + 1);
  for (int c1 = max(1, -c0 + 9); c1 <= min(c0 - 4, -c0 + 12); c1 += 1)
    s0(c1, c0 + c1 - 8);
  for (int c1 = max(c0 - 3, -c0 + 9); c1 <= min(c0, -c0 + 12); c1 += 1) {
    s0(c1, c0 + c1 - 8);
    s1(c1, c0 - c1 + 1);
  }
  for (int c1 = max(c0 - 3, -c0 + 13); c1 <= min(8, c0); c1 += 1)
    s1(c1, c0 - c1 + 1);
  for (int c1 = max(c0 + 1, -c0 + 9); c1 <= min(8, -c0 + 12); c1 += 1)
    s0(c1, c0 + c1 - 8);
}
