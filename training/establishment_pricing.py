"""
============================================================
FICHIER 1 : pricing/serializers/establishment_pricing.py
Serializers pour PriceRule et SpecialPrice côté établissement
============================================================
"""
from rest_framework import serializers
from pricing.models import PriceRule, SpecialPrice


class EstablishmentPriceRuleSerializer(serializers.ModelSerializer):
    """Lecture d'une PriceRule"""

    rule_type_display = serializers.CharField(
        source='get_rule_type_display',
        read_only=True
    )

    class Meta:
        model = PriceRule
        fields = [
            'id',
            'asset_type', 'asset_id', 'option',
            'rule_type', 'rule_type_display',
            'price_override', 'discount_percent', 'discount_amount',
            'currency',
            'date_start', 'date_end',
            'weekdays',
            'min_stay', 'max_stay',
            'priority', 'active',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'asset_type', 'asset_id', 'created_at', 'updated_at']


class EstablishmentPriceRuleCreateSerializer(serializers.ModelSerializer):
    """Création/modification d'une PriceRule"""

    class Meta:
        model = PriceRule
        fields = [
            'asset_id', 'option',
            'rule_type',
            'price_override', 'discount_percent', 'discount_amount',
            'currency',
            'date_start', 'date_end',
            'weekdays',
            'min_stay', 'max_stay',
            'priority', 'active',
        ]

    def validate(self, data):
        rule_type = data.get('rule_type')

        # Vérifier cohérence rule_type / valeur
        if rule_type == 'OVERRIDE' and not data.get('price_override'):
            raise serializers.ValidationError({
                'price_override': "Requis pour le type OVERRIDE"
            })

        if rule_type == 'DISCOUNT_PERCENT' and not data.get('discount_percent'):
            raise serializers.ValidationError({
                'discount_percent': "Requis pour le type DISCOUNT_PERCENT"
            })

        if rule_type == 'DISCOUNT_AMOUNT' and not data.get('discount_amount'):
            raise serializers.ValidationError({
                'discount_amount': "Requis pour le type DISCOUNT_AMOUNT"
            })

        # Dates cohérentes
        if data.get('date_start') and data.get('date_end'):
            if data['date_end'] < data['date_start']:
                raise serializers.ValidationError({
                    'date_end': "La date de fin doit être après la date de début"
                })

        # min_stay / max_stay cohérents
        if data.get('min_stay') and data.get('max_stay'):
            if data['max_stay'] < data['min_stay']:
                raise serializers.ValidationError({
                    'max_stay': "max_stay doit être supérieur à min_stay"
                })

        # Valider asset_id appartient à l'établissement
        asset_id = data.get('asset_id')
        if asset_id:
            establishment_uuid = self.context['view'].kwargs.get('establishment_uuid')
            from accommodations.models import Accommodation
            if not Accommodation.objects.filter(
                id=asset_id,
                establishment__uuid=establishment_uuid
            ).exists():
                raise serializers.ValidationError({
                    'asset_id': "Cet hébergement n'appartient pas à votre établissement"
                })

        # Valider option appartient à l'asset
        option_id = data.get('option')
        if option_id and asset_id:
            from accommodations.models import AccommodationOption
            if not AccommodationOption.objects.filter(
                id=option_id,
                accommodation_id=asset_id
            ).exists():
                raise serializers.ValidationError({
                    'option': "Cette option n'appartient pas à cet hébergement"
                })

        return data

    def create(self, validated_data):
        # asset_type toujours ACCOMMODATION côté établissement
        validated_data['asset_type'] = 'ACCOMMODATION'
        return super().create(validated_data)


class EstablishmentSpecialPriceSerializer(serializers.ModelSerializer):
    """Lecture d'un SpecialPrice"""

    class Meta:
        model = SpecialPrice
        fields = [
            'id',
            'asset_type', 'asset_id', 'option',
            'date', 'price', 'currency',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'asset_type', 'created_at', 'updated_at']


class EstablishmentSpecialPriceCreateSerializer(serializers.ModelSerializer):
    """Création/modification d'un SpecialPrice"""

    class Meta:
        model = SpecialPrice
        fields = [
            'asset_id', 'option',
            'date', 'price', 'currency',
        ]

    def validate(self, data):
        # Valider asset_id appartient à l'établissement
        asset_id = data.get('asset_id')
        if asset_id:
            establishment_uuid = self.context['view'].kwargs.get('establishment_uuid')
            from accommodations.models import Accommodation
            if not Accommodation.objects.filter(
                id=asset_id,
                establishment__uuid=establishment_uuid
            ).exists():
                raise serializers.ValidationError({
                    'asset_id': "Cet hébergement n'appartient pas à votre établissement"
                })

        # Valider option appartient à l'asset
        option_id = data.get('option')
        if option_id and asset_id:
            from accommodations.models import AccommodationOption
            if not AccommodationOption.objects.filter(
                id=option_id,
                accommodation_id=asset_id
            ).exists():
                raise serializers.ValidationError({
                    'option': "Cette option n'appartient pas à cet hébergement"
                })

        # Vérifier unicité (asset_type, asset_id, option, date)
        if self.instance is None:  # création seulement
            if SpecialPrice.objects.filter(
                asset_type='ACCOMMODATION',
                asset_id=asset_id,
                option=option_id,
                date=data.get('date')
            ).exists():
                raise serializers.ValidationError(
                    "Un prix spécial existe déjà pour cet hébergement/option à cette date"
                )

        return data

    def create(self, validated_data):
        validated_data['asset_type'] = 'ACCOMMODATION'
        return super().create(validated_data)


"""
============================================================
FICHIER 2 : pricing/views/establishment_pricing.py
ViewSets PriceRule et SpecialPrice côté établissement
============================================================
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend

from pricing.models import PriceRule, SpecialPrice
from pricing.serializers.establishment_pricing import (
    EstablishmentPriceRuleSerializer,
    EstablishmentPriceRuleCreateSerializer,
    EstablishmentSpecialPriceSerializer,
    EstablishmentSpecialPriceCreateSerializer,
)


class EstablishmentPriceRuleViewSet(viewsets.ModelViewSet):
    """
    Gestion des règles de prix côté établissement.

    GET    /establishments/{uuid}/price-rules/
    POST   /establishments/{uuid}/price-rules/
    GET    /establishments/{uuid}/price-rules/{id}/
    PATCH  /establishments/{uuid}/price-rules/{id}/
    DELETE /establishments/{uuid}/price-rules/{id}/

    Filtres disponibles :
    - ?asset_id=1       → règles d'un hébergement
    - ?option=1         → règles d'une option
    - ?active=true      → règles actives uniquement
    - ?rule_type=OVERRIDE
    """

    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['asset_id', 'option', 'active', 'rule_type']
    http_method_names = ['get', 'post', 'patch', 'delete']

    def get_queryset(self):
        establishment_uuid = self.kwargs['establishment_uuid']

        # Récupérer tous les asset_ids de l'établissement
        from accommodations.models import Accommodation
        asset_ids = Accommodation.objects.filter(
            establishment__uuid=establishment_uuid
        ).values_list('id', flat=True)

        return PriceRule.objects.filter(
            asset_type='ACCOMMODATION',
            asset_id__in=asset_ids
        ).order_by('priority', '-created_at')

    def get_serializer_class(self):
        if self.action in ['create', 'partial_update', 'update']:
            return EstablishmentPriceRuleCreateSerializer
        return EstablishmentPriceRuleSerializer

    @action(detail=True, methods=['post'], url_path='toggle')
    def toggle_active(self, request, establishment_uuid=None, pk=None):
        """Active ou désactive une règle"""
        rule = self.get_object()
        rule.active = not rule.active
        rule.save(update_fields=['active'])

        return Response({
            'id':     rule.id,
            'active': rule.active,
            'message': f"Règle {'activée' if rule.active else 'désactivée'}"
        })


class EstablishmentSpecialPriceViewSet(viewsets.ModelViewSet):
    """
    Gestion des prix spéciaux côté établissement.

    GET    /establishments/{uuid}/special-prices/
    POST   /establishments/{uuid}/special-prices/
    GET    /establishments/{uuid}/special-prices/{id}/
    PATCH  /establishments/{uuid}/special-prices/{id}/
    DELETE /establishments/{uuid}/special-prices/{id}/

    Filtres disponibles :
    - ?asset_id=1       → prix d'un hébergement
    - ?option=1         → prix d'une option
    - ?date=2026-12-31  → prix d'une date précise
    """

    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['asset_id', 'option', 'date']
    http_method_names = ['get', 'post', 'patch', 'delete']

    def get_queryset(self):
        establishment_uuid = self.kwargs['establishment_uuid']

        from accommodations.models import Accommodation
        asset_ids = Accommodation.objects.filter(
            establishment__uuid=establishment_uuid
        ).values_list('id', flat=True)

        return SpecialPrice.objects.filter(
            asset_type='ACCOMMODATION',
            asset_id__in=asset_ids
        ).order_by('-date')

    def get_serializer_class(self):
        if self.action in ['create', 'partial_update', 'update']:
            return EstablishmentSpecialPriceCreateSerializer
        return EstablishmentSpecialPriceSerializer


"""
============================================================
FICHIER 3 : establishments/urls.py
Ajouter les nouvelles routes
============================================================
"""

# Ajouter dans ton router establishments :

# from pricing.views.establishment_pricing import (
#     EstablishmentPriceRuleViewSet,
#     EstablishmentSpecialPriceViewSet,
# )

# router.register(
#     r'establishments/(?P<establishment_uuid>[^/.]+)/price-rules',
#     EstablishmentPriceRuleViewSet,
#     basename='establishment-price-rules'
# )

# router.register(
#     r'establishments/(?P<establishment_uuid>[^/.]+)/special-prices',
#     EstablishmentSpecialPriceViewSet,
#     basename='establishment-special-prices'
# )
