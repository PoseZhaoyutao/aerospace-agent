from __future__ import annotations

from base64 import b64encode
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aerospace_agent.langgraph_agent.agent_core.approval import (
    CapabilityApprovalLedger,
    CapabilityApprovalVerifier,
    approval_signature_payload,
)


def _keys():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private, public


def test_external_signature_is_persisted_and_verified_without_private_key(tmp_path) -> None:
    private, public = _keys()
    digest = "a" * 64
    signature = private.sign(approval_signature_payload(digest))
    ledger = CapabilityApprovalLedger(
        tmp_path / "approval.sqlite", trusted_public_keys={"operator-1": public}
    )

    record = ledger.append(
        approval_record_id="approval-1",
        key_id="operator-1",
        digest=digest,
        signature_b64=b64encode(signature).decode("ascii"),
        created_at=datetime(2026, 7, 13, tzinfo=UTC).isoformat(),
    )

    assert ledger.verify(digest) == record
    assert CapabilityApprovalVerifier(ledger).verify_digest(digest) == record
    assert ledger.schema_version() == 1


def test_invalid_signature_unknown_key_and_digest_are_rejected(tmp_path) -> None:
    private, public = _keys()
    ledger = CapabilityApprovalLedger(
        tmp_path / "approval.sqlite", trusted_public_keys={"operator-1": public}
    )

    with pytest.raises(ValueError, match="signature"):
        ledger.append(
            approval_record_id="bad",
            key_id="operator-1",
            digest="a" * 64,
            signature_b64=b64encode(private.sign(approval_signature_payload("b" * 64))).decode(),
            created_at=datetime(2026, 7, 13, tzinfo=UTC).isoformat(),
        )
    with pytest.raises(ValueError, match="trusted key"):
        ledger.append(
            approval_record_id="unknown",
            key_id="other",
            digest="a" * 64,
            signature_b64=b64encode(b"x" * 64).decode(),
            created_at=datetime(2026, 7, 13, tzinfo=UTC).isoformat(),
        )
    assert ledger.verify("b" * 64) is None


def test_approval_records_are_append_only_and_revocation_is_a_separate_event(tmp_path) -> None:
    private, public = _keys()
    ledger = CapabilityApprovalLedger(
        tmp_path / "approval.sqlite", trusted_public_keys={"operator-1": public}
    )
    digest = "c" * 64
    signature = b64encode(private.sign(approval_signature_payload(digest))).decode()
    ledger.append(
        approval_record_id="approval-1",
        key_id="operator-1",
        digest=digest,
        signature_b64=signature,
        created_at=datetime(2026, 7, 13, tzinfo=UTC).isoformat(),
    )
    with pytest.raises(ValueError, match="append-only"):
        ledger.append(
            approval_record_id="approval-1",
            key_id="operator-1",
            digest=digest,
            signature_b64=signature,
            created_at=datetime(2026, 7, 13, tzinfo=UTC).isoformat(),
        )

    ledger.revoke(
        approval_record_id="approval-1",
        revoked_at=datetime(2026, 7, 14, tzinfo=UTC).isoformat(),
        reason="validation baseline withdrawn",
    )

    assert ledger.verify(digest) is None
    assert ledger.events("approval-1")[-1]["event_type"] == "revoked"
