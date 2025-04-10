#include "maskElement.h"
#include <cstdint>

int MaskElement::getIndex() { return this->index; }

bool MaskElement::isFree() { return this->free; }

uint64_t MaskElement::getMask() { return this->TPCMask; }

void MaskElement::disable() { this->free = false; }

void MaskElement::enable() { this->free = true; }
